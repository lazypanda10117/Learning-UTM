import numpy as np

from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
from models import transformer

def test_utm():
    rng = np.random.default_rng(seed=2)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler, alphabet_size=16)
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

    params = data_generator.sample_params(sample_size=data_generator.batch_size)
    sequences, categorical_probs, extra = data_generator.sample_from_params(params=params)
    # print("Sequences: ", sequences)
    # print("Categorical Probs: ", categorical_probs)
    # print("Extra: ", extra)
    log_dict = {
        "categorical_probs": categorical_probs,
        "params": params,
    }
    log_dict.update(extra)
    # print(log_dict)
    total_non_zero_length = 0
    for sequence in sequences:
        # print("sequence shape: ", sequence.shape)
        # print("sequence: ", sequence)
        print("sequence: ", sequence.shape)
        indices = [int(np.where(token == 1)[0][0]) for token in sequence]
        print(''.join([str(idx) + ' ' * (4 - len(str(idx))) for idx in indices]))
        total_non_zero_length += len(list(filter(lambda x: x != 0, indices)))
    print("Total non-zero length: ", total_non_zero_length)
    print("Average non-zero length: ", total_non_zero_length / len(sequences))

test_utm()