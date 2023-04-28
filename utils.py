from config import embed_kwargs, general_kwargs
from embedder import Embedder
from nerf import NeRF
import torch
import torch.nn as nn
import numpy as np

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def get_embedder(i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Helps GPU to process big amounts of data
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

# Model for movies does not assume viewdirs
# 65536 = 1024*64
def run_network(inputs, fn, embed_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(device):
    embed_fn, input_ch = get_embedder(general_kwargs['multires'], general_kwargs['i_embed'])
    output_ch = 3 # RGB at given (x,y,t)
    skips = [4]

    model = NeRF(D=general_kwargs['D'], W=general_kwargs['W'],
                 input_ch=input_ch, output_ch=output_ch, skips=skips).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, network_fn : run_network(inputs, network_fn,
                                                                embed_fn=embed_fn,
                                                                netchunk=1024*64)
    optimizer = torch.optim.Adam(params=grad_vars, lr=general_kwargs['lrate'], betas=(0.9, 0.999))

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_fn' : model,
        'optimizer': optimizer,
        'grad_vars': grad_vars,
    }

    return render_kwargs_train

