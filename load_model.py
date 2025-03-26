import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from models import transformer
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
import functools


def make_config_generator(
    vocab_size: int,
):
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=6,
        embedding_dim=256,
        num_heads=4,
    )


def load_model_params(
    data_generator, params_path: str, vocab_size: int
) -> tuple[hk.Transformed, hk.Params]:
    """Loads saved model parameters and returns the initialized model and params.

    Args:
        params_path: Path to the saved .npz file containing model parameters
        vocab_size: Size of the vocabulary used by the model

    Returns:
        A tuple containing (model, params) where model is the initialized Haiku model
        and params are the loaded parameters
    """
    # Create the same model configuration as used in training
    config = make_config_generator(vocab_size)
    model: hk.Transformed = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )

    # Load the saved parameters
    params = np.load(params_path, allow_pickle=True)
    print(params.items())

    # Initialize the model with a dummy batch to get the parameter structure
    dummy_batch, _ = data_generator.sample_dummy(
        data_generator.batch_size
    )  # Minimal dummy input
    rng = jax.random.PRNGKey(0)
    print("got here")
    model.init(rng, dummy_batch)
    print("but not here")

    # Convert numpy arrays to JAX arrays and ensure parameter structure matches
    params = jax.tree_map(jnp.array, params)

    return model, params


def main():
    # Example usage
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

    # Load the model and parameters
    model, params = load_model_params(
        data_generator, "params.npz", data_generator.feature_size
    )

    # Example of using the loaded model
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)  # Example input
    predictions = model.apply(params, dummy_input, rng=None)
    print("Predictions: ", predictions)
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
