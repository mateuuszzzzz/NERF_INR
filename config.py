import torch

general_kwargs = {
    'multires': 10,
    'i_embed': 0,
    'D': 8,
    'W': 256,
    'epochs': 100,
    'lrate': 0.0005
}

embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : general_kwargs['multires']-1,
                'num_freqs' : general_kwargs['multires'],
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
}