# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a neural model on some data generated from the data/ folder."""

from cProfile import label
import functools
from turtle import color
from typing import Any

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

from data import data_generator as dg_lib
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
from models import transformer

import matplotlib.pyplot as plt

from data import (
    chomsky_data_generator as chomsky_sampler_lib,
)


def make_config_generator(
    vocab_size: int,
):
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=6,
        embedding_dim=256,
        num_heads=4,
    )


def _make_loss_fn(model: hk.Transformed) -> Any:
    """Returns the loss function for update_parameters."""

    def loss_fn(
        params: hk.Params,
        sequences: jax.Array,
        mask: jax.Array,
    ) -> jnp.float32:
        """Returns the loss for the model and the last state.

        Args:
          params: The parameters of the model, usually a neural network.
          sequences: The input of sequences to evaluate. See neural_predictors.py.
          mask: A binary array, True (1's) denote where to skip computing the loss.
        """
        # This code computes the loss for a transformer decoder model:
        # 1. Apply the model to get log probabilities (conditionals) for each token
        conditionals = model.apply(
            params=params,
            targets=sequences,
            rng=None,
        )
        # 2. Extract the log probabilities of the actual tokens that appeared in the sequence
        # by using take_along_axis to select the probability corresponding to each token
        true_conditionals = jnp.take_along_axis(
            conditionals, sequences[..., None], axis=-1
        )[..., 0]
        # 3. Apply the mask to zero out log probabilities where we should skip computing loss (e.g., for padding tokens)
        true_conditionals = jnp.where(mask, 0.0, true_conditionals)
        # 4. Sum the log probabilities across the sequence dimension to get log likelihood per batch
        marginals = jnp.sum(true_conditionals, axis=1)  # Shape (B,).
        # 5. Return the negative mean log likelihood as the loss (for minimization)
        return -jnp.mean(marginals)

    return loss_fn


@functools.partial(
    jax.jit, static_argnames=("optimizer", "grad_fn", "normalize_gradients")
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    mask: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
    """Returns updated params and extra logs (like loss, last state etc).

    Backpropagation is done on the whole sequence. The whole function is jitted.

    Args:
      params: The current parameters of the network.
      opt_state: The optimizer state.
      sequences: The input of sequences to evaluate. See base_predictor.py.
      mask: A binary array, True (1's) denote where to skip computing the loss.
      grad_fn: A gradient function, which takes some parameters, a random seed,
        the data to compute the gradient on, and an initial state for the
        predictor. It returns the gradient of the parameters for this batch of
        data, and extra values.
      optimizer: An optax optimizer.
      normalize_gradients: Whether to divide the gradients by the length of the
        sequences, or keep them as is. Using this option guarantees to have the
        same scale across various sequence lengths, and therefore tasks.
    """
    loss, grad = grad_fn(params, sequences, mask)
    if normalize_gradients:
        length_sequence = float(sequences.shape[1])
        grad = tree.map_structure(lambda x: x / length_sequence, grad)
    updates, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)

    log_dict = {
        "loss": loss,
        "grad_norm_unclipped": optax.global_norm(grad),
    }

    return new_params, new_opt_state, log_dict


def train_transformer_decoder(
    data_generator: dg_lib.DataGenerator,
    training_steps: int,
    log_every: int,
    batch_size: int = 128,
    use_tqdm: bool = True,
    with_markov: bool = False,
    eval_data_generator: dg_lib.DataGenerator = None,
) -> tuple[hk.Params, float, list[float], list[float], list[float]]:
    """Trains a neural network on some synthetic data.

    We train a decoder-only transformer on batches, minimizing the log-loss
    objective. The exact architecture can be modified using the TransformerConfig
    object (defined in models/transformer.py)

    Args:
      data_generator: Used to generate batches of data to train on.
      training_steps: Number of batches to train on.
      log_every: How often to log the loss. If negative or 0, no log at all.
      batch_size: The number of sequences in a batch.
      use_tqdm: Whether to use a progress bar or not.

    Returns:
      The final loss, and final parameters.
    """
    print("Vocab size:", data_generator.feature_size)
    config = make_config_generator(data_generator.feature_size)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )

    # Initialize parameters.
    dummy_batch, _ = data_generator.sample_dummy(batch_size)
    # Transform one-hots to integer tokens.
    dummy_batch = np.argmax(dummy_batch, axis=-1)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_batch)

    # Make gradient function.
    loss_fn = _make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    # Make optimizer, to apply the gradients.
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    logging.info("Initialization done, starting training...")
    last_loss = 0.0
    default_mask = lambda x: np.ones(x.shape[:2], dtype=bool)
    eval_losses = []
    eval_accs = []
    eval_final_accs = []
    for step in tqdm.trange(training_steps, disable=not use_tqdm):
        batch, log_dict = data_generator.sample(with_markov=True)
        # Transform one-hots to integer tokens.
        batch = np.argmax(batch, axis=-1)
        if "loss_mask" in log_dict:
            loss_mask = log_dict["loss_mask"]
        else:
            loss_mask = default_mask(batch)

        params, opt_state, logs = _update_parameters(
            params=params,
            opt_state=opt_state,
            sequences=batch,
            grad_fn=grad_fn,
            optimizer=optimizer,
            mask=loss_mask,
        )
        if log_every > 0 and step % log_every == 0:
            logging.info(
                "Step %d, Loss (avg cumulative nats) %f, Grad norm %f",
                step,
                logs["loss"],
                logs["grad_norm_unclipped"],
            )
        if step % 100 == 0:
            last_loss = logs["loss"]
            eval_loss, eval_acc, eval_final_acc = evaluate_transformer_decoder(
                eval_data_generator, params, data_generator
            )
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            eval_final_accs.append(eval_final_acc)

    return params, last_loss, eval_losses, eval_accs, eval_final_accs


def evaluate_transformer_decoder(
    data_generator: dg_lib.DataGenerator,
    params: hk.Params,
    training_data_generator: dg_lib.DataGenerator = None,
    num_batches: int = 10,
) -> tuple[float, float, float]:
    """Evaluates a neural network on some synthetic data.

    We evaluate a decoder-only transformer on batches, minimizing the log-loss
    objective. The exact architecture can be modified using the TransformerConfig
    object (defined in models/transformer.py)
    """

    config = make_config_generator(training_data_generator.feature_size)
    model: hk.Transformed = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )

    # Initialize parameters.
    dummy_batch, _ = data_generator.sample_dummy(data_generator.batch_size)
    # Transform one-hots to integer tokens.
    dummy_batch = np.argmax(dummy_batch, axis=-1)
    rng = jax.random.PRNGKey(0)
    regret = 0.0
    default_mask = lambda x: np.zeros(x.shape[:2], dtype=bool)
    total_accuracy = 0.0
    total_final_accuracy = 0.0

    for i in range(num_batches):
        batch, log_dict = data_generator.sample()
        batch = np.argmax(batch, axis=-1)
        if "input_locations" in log_dict:
            input_mask = log_dict["input_locations"]
        else:
            input_mask = default_mask(batch)
        conditionals = model.apply(
            params=params,
            targets=batch,
            rng=None,
        )
        true_conditionals = jnp.take_along_axis(
            conditionals, batch[..., None], axis=-1
        )[..., 0]
        accuracy = jnp.exp(true_conditionals)
        true_accuracies = accuracy[~input_mask]
        true_conditionals = jnp.where(input_mask, 0.0, true_conditionals)
        avg_accuracy = jnp.mean(true_accuracies)
        final_accuracy = jnp.mean(true_accuracies[:, -1])
        total_accuracy += avg_accuracy

        marginals = jnp.sum(true_conditionals, axis=1)  # Shape (B,).
        total_final_accuracy += final_accuracy
        regret += -jnp.mean(marginals)

    regret /= num_batches
    total_accuracy /= num_batches
    total_final_accuracy /= num_batches

    return regret, total_accuracy, total_final_accuracy


def main(_) -> None:
    """Trains a model and save the parameters to a file."""

    TRAINING_STEPS = 200

    rng = np.random.default_rng(seed=1)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    data_generator = utm_dg_lib.UTMDataGenerator(
        batch_size=32,
        seq_length=256,
        rng=rng,
        utm=utm,
        memory_size=10,
        maximum_steps=200,
        tokenizer=utm_dg_lib.Tokenizer.ASCII,
        maximum_program_length=100,
    )

    chomsky_generator = chomsky_sampler_lib.ChomskyDataGenerator(
        task_str="even_pairs",
        max_input_length=20,
        use_delimiters=True,
        batch_size=32,
        seq_length=256,
        rng=rng,
    )

    params, loss, eval_losses, eval_accs, eval_final_accs = train_transformer_decoder(
        data_generator=data_generator,
        training_steps=TRAINING_STEPS,
        log_every=10,
        with_markov=True,
        eval_data_generator=chomsky_generator,
    )
    logging.info("Final loss: %f", loss)

    np.savez("params.npz", **params)
    logging.info("Parameters saved in file params.npz")

    plt.plot(eval_accs, label="avg", color="red")
    plt.plot(eval_final_accs, label="final", color="blue")
    plt.legend()
    plt.show()

    evaluate_transformer_decoder(
        chomsky_generator, params, training_data_generator=data_generator
    )

    # for _ in range(10):
    #     print("batch")
    #     print()
    #     batch, _ = chomsky_generator.sample()
    #     print(batch)
    #     print()

    # chomsky_params, chomsky_loss = train_transformer_decoder(
    #     data_generator=chomsky_generator,
    #     training_steps=1000,
    #     log_every=10,
    # )


if __name__ == "__main__":
    app.run(main)
