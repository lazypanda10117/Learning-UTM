from models import transformer
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
import haiku as hk
import functools
import jax
import numpy as np
from data import (
    chomsky_data_generator as chomsky_sampler_lib,
)
from data import data_generator as dg_lib
import jax.numpy as jnp


def make_config_generator(
    vocab_size: int,
):
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=6,
        embedding_dim=256,
        num_heads=4,
    )


def utm_data_generator(rng):
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)

    return utm_dg_lib.UTMDataGenerator(
        batch_size=32,
        seq_length=256,
        rng=rng,
        utm=utm,
        memory_size=10,
        maximum_steps=200,
        tokenizer=utm_dg_lib.Tokenizer.ASCII,
        maximum_program_length=100,
    )


def make_chomsky_generator(rng):
    return chomsky_sampler_lib.ChomskyDataGenerator(
        task_str="even_pairs",
        max_input_length=20,
        use_delimiters=True,
        batch_size=32,
        seq_length=256,
        rng=rng,
    )


def make_model(data_generator):
    config = make_config_generator(data_generator.feature_size)
    return hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )


def init_params(model, data_generator, batch_size):
    dummy_batch, _ = data_generator.sample_dummy(batch_size)
    # Transform one-hots to integer tokens.
    dummy_batch = np.argmax(dummy_batch, axis=-1)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_batch)
    return params


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

    model = make_model(training_data_generator)
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
