from config import general_kwargs
from nerf_utils import create_nerf, img2mse, construct_images_with_model
from prepare_dataset import PixelDataLoader, PixelDataset
import torch
import os

def train_loop():

    # get envs
    dataset_path = os.getenv('DATASET_PATH')
    save_path = os.getenv('SAVE_PATH')
    frames = int(os.getenv('FRAMES'))

    # Prepare model for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_kwargs_train = create_nerf(device)

    network_query_fn = render_kwargs_train['network_query_fn']
    network_fn = render_kwargs_train['network_fn']
    optimizer = render_kwargs_train['optimizer']
    # grad_vars = render_kwargs_train['grad_vars']

    train_imgs_folder = os.path.join(dataset_path, 'train')
    train_dataset = PixelDataset(train_imgs_folder)
    train_dataloader = PixelDataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)

    val_imgs_folder = os.path.join(dataset_path, 'val')
    val_dataset = PixelDataset(val_imgs_folder)
    val_dataloader = PixelDataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=4)

    for epoch in range(general_kwargs['epochs']):

        print("Epoch: ", epoch)

        for idx, batch in enumerate(train_dataloader):
            print("Batch: ", idx, "/", len(train_dataloader))
            inputs, targets = batch['position'], batch['rgb']
            optimizer.zero_grad()

            outputs = network_query_fn(inputs, network_fn)
            loss = img2mse(outputs, targets)
            
            loss.backward()
            optimizer.step()

        # Validation stage
        total_loss = 0
        for batch in val_dataloader:
            inputs, targets = batch['position'], batch['rgb']

            outputs = network_query_fn(inputs, network_fn)
            total_loss += img2mse(outputs, targets)
        print("Total loss for validation: ", total_loss)

    
    # Create a movie from the trained model
    construct_images_with_model(network_fn, save_path, dataset_path)

if __name__ == '__main__':
    train_loop()