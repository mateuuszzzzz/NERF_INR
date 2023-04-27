from config import embed_kwargs, general_kwargs
from embedder import Embedder
from nerf import NeRF
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

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


def network_query_fn(inputs, model):
    embed_fn, input_ch = get_embedder(general_kwargs['multires'], general_kwargs['i_embed'])

    model = NeRF(D=general_kwargs['D'], W=general_kwargs['W'],
                 input_ch=input_ch, output_ch=3, skips=[4]).to(device)
    
    return run_network(inputs, model, embed_fn)