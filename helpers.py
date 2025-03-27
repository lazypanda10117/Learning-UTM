import haiku as hk
import functools
import jax
import jax.numpy as jnp
import numpy as np

from data import (
    chomsky_data_generator as chomsky_sampler_lib,
)
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
from models import transformer

CHOMSKY_ALPHABET_SIZE = 17


def save_params(params, filename):
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    np.savez(filename, *flat_params, tree_def=tree_def)


def make_transformer_config(
    vocab_size: int,
    size: str = "large",
):
    if size == "large":
        return transformer.TransformerConfig(
            vocab_size=vocab_size,
            num_layers=6,
            embedding_dim=256,
            num_heads=4,
        )
    elif size == "medium":
        return transformer.TransformerConfig(
            vocab_size=vocab_size,
            num_layers=4,
            embedding_dim=64,
            num_heads=4,
        )
    elif size == "small":
        return transformer.TransformerConfig(
            vocab_size=vocab_size,
            num_layers=2,
            embedding_dim=16,
            num_heads=2,
        )
    else:
        raise ValueError(f"Invalid transformer size: {size}")


def utm_data_generator(
    rng,
    maximum_steps=200,
    seq_length=256,
    maximum_program_length=100,
    memory_size=10,
    alphabet_size=CHOMSKY_ALPHABET_SIZE,
    batch_size=32,
):
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler, alphabet_size=alphabet_size)

    return utm_dg_lib.UTMDataGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng,
        utm=utm,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        tokenizer=utm_dg_lib.Tokenizer.ASCII,
        maximum_program_length=maximum_program_length,
    )


def make_chomsky_generator(
    rng,
    task_str="even_pairs",
    use_delimiters=True,
    max_input_length=None,
    batch_size=40,
):

    if max_input_length is None:
        max_input_length = 32 if task_str == "compute_sqrt" else 256

    return chomsky_sampler_lib.ChomskyDataGenerator(
        task_str=task_str,
        max_input_length=max_input_length,
        use_delimiters=use_delimiters,
        batch_size=batch_size,
        seq_length=256,
        expand_feature_size=CHOMSKY_ALPHABET_SIZE - 2,
        rng=rng,
    )


def make_model(data_generator, size: str = "large"):
    config = make_transformer_config(data_generator.feature_size, size)
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
    chomsky_data_generator: chomsky_sampler_lib.ChomskyDataGenerator,
    params: hk.Params,
    training_data_generator: utm_dg_lib.UTMDataGenerator = None,
    num_batches: int = 10,
    size: str = "large",
) -> tuple[float, float, float]:
    """Evaluates a neural network on some synthetic data.

    We evaluate a decoder-only transformer on batches, minimizing the log-loss
    objective. The exact architecture can be modified using the TransformerConfig
    object (defined in models/transformer.py)
    """
    print("Chomsky Task: ", chomsky_data_generator._task_str)

    model = make_model(training_data_generator, size)
    regret = 0.0
    default_mask = lambda x: np.zeros(x.shape[:2], dtype=bool)
    total_accuracy = 0.0
    total_final_accuracy = 0.0

    for _ in range(num_batches):
        batch, log_dict = chomsky_data_generator.sample()
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
