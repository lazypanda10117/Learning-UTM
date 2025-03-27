import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from helpers import (
    evaluate_transformer_decoder,
    make_chomsky_generator,
    make_model,
    utm_data_generator,
)
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
    data_generator, params_path: str, vocab_size: int, batch_size: int = 128
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
    model = make_model(data_generator)

    # Load the saved parameters
    loaded = np.load("params.npz", allow_pickle=True)
    tree_def = loaded["tree_def"].item()  # Get PyTreeDef
    flat_params = [loaded[f"arr_{i}"] for i in range(len(loaded.files) - 1)]
    loaded_params = jax.tree_util.tree_unflatten(tree_def, flat_params)

    # Initialize the model with a dummy batch to get the parameter structure
    dummy_batch, _ = data_generator.sample_dummy(128)  # Minimal dummy input
    dummy_batch = np.argmax(dummy_batch, axis=-1)

    rng = jax.random.PRNGKey(0)
    model.init(rng, dummy_batch)

    return model, loaded_params


def main():
    # Example usage
    rng = np.random.default_rng(seed=1)
    data_generator = utm_data_generator(rng)
    chomsky_generator = make_chomsky_generator(rng)

    # Load the model and parameters
    model, params = load_model_params(
        data_generator, "params.npz", data_generator.feature_size
    )

    regret, total_accuracy, total_final_accuracy = evaluate_transformer_decoder(
        chomsky_generator, params, data_generator
    )
    print(total_accuracy, total_final_accuracy)


if __name__ == "__main__":
    main()
