import torch

embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : 10-1,
                'num_freqs' : 10,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
}