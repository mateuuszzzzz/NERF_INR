from config import embed_kwargs
from embedder import Embedder
import torch.nn as nn


def get_embedder(i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim