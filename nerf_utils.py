from config import embed_kwargs, general_kwargs
from embedder import Embedder
from nerf import NeRF
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def get_embedder(i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Model for movies does not assume viewdirs
# 65536 = 1024*64
def run_network(inputs, fn, embed_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = fn(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(device):
    embed_fn, input_ch = get_embedder()
    output_ch = 3 # RGB at given (x,y,t)
    skips = [4]

    model = NeRF(D=general_kwargs['D'], W=general_kwargs['W'],
                 input_ch=input_ch, output_ch=3, skips=skips).to(device)
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

def construct_images_with_model(model, save_path, dataset_path):
        

        def construct_image(frame, save_path, dataset_path):
            # get width and height of real image
            real_image = Image.open(f'{dataset_path}/frame_{frame}.png')
            width, height = real_image.size

            generated_image = Image.new('RGB', (width, height))

            for x in range(width):
                for y in range(height):
                    # get pixel color from model
                    pixel_color = model(torch.Tensor([x, y, frame]))
                    # set pixel color in real image
                    generated_image.putpixel((x, y), tuple(pixel_color))
            
            # model generated image
            generated_image.save(f'{save_path}/frame_{frame}_generated.png')

        train_dataset_path = os.path.join(dataset_path, 'train')
        val_dataset_path = os.path.join(dataset_path, 'val')

        train_imgs = os.listdir(train_dataset_path)

        for train_img_pat in train_imgs:
            frame = int(train_img_pat.split('_')[1].split('.')[0])
            construct_image(frame, save_path, dataset_path)

        for val_img_pat in os.listdir(val_dataset_path):
            frame = int(val_img_pat.split('_')[1].split('.')[0])
            construct_image(frame, save_path, dataset_path)