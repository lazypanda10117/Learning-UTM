import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from neural_networks_solomonoff_induction.models import transformer
from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg_lib
from neural_networks_solomonoff_induction.data import utms as utms_lib

def load_model_params(params_path: str, vocab_size: int) -> tuple[hk.Transformed, hk.Params]:
    """Loads saved model parameters and returns the initialized model and params.
    
    Args:
        params_path: Path to the saved .npz file containing model parameters
        vocab_size: Size of the vocabulary used by the model
        
    Returns:
        A tuple containing (model, params) where model is the initialized Haiku model
        and params are the loaded parameters
    """
    # Create the same model configuration as used in training
    config = transformer.TransformerConfig(vocab_size=vocab_size)
    model = hk.transform(
        lambda x: transformer.transformer_decoder(x, config=config)
    )
    
    # Load the saved parameters
    params = np.load(params_path)
    
    # Initialize the model with a dummy batch to get the parameter structure
    dummy_batch = jnp.zeros((1, 1), dtype=jnp.int32)  # Minimal dummy input
    rng = jax.random.PRNGKey(0)
    init_params = model.init(rng, dummy_batch)
    
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
        maximum_steps=100,
        tokenizer=utm_dg_lib.Tokenizer.ASCII,
        maximum_program_length=100,
    )
    
    # Load the model and parameters
    model, params = load_model_params('params.npz', data_generator.feature_size)
    
    # Example of using the loaded model
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)  # Example input
    predictions = model.apply(params, dummy_input, rng=None)
    print("Predictions: ", predictions)
    print("Model loaded successfully!")

if __name__ == '__main__':
    main() 