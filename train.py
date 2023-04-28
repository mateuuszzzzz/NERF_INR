from config import general_kwargs
from utils import create_nerf, img2mse
from prepare_dataset import CustomDataloader, ImageDataset
import torch

def train_loop():
    # Prepare model for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_kwargs_train = create_nerf(device)

    network_query_fn = render_kwargs_train['network_query_fn']
    network_fn = render_kwargs_train['network_fn']
    optimizer = render_kwargs_train['optimizer']
    grad_vars = render_kwargs_train['grad_vars']

    img_folder = 'path/to/images/folder'
    dataset = ImageDataset(img_folder)
    dataloader = CustomDataloader(dataset, batch_size=10, shuffle=True, num_workers=8)

    for epoch in range(general_kwargs['epochs']):
        
        # Here probably should be some loss variable

        for batch in dataloader:
            inputs, targets = batch

            optimizer.zero_grad()

            outputs = network_query_fn(inputs)
            loss = img2mse(outputs, targets)
            
            loss.backwards()
            optimizer.step()

    print("SUCCESS")


if __name__ == '__main__':
    train_loop()