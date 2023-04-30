import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class PixelDataset(Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_list = sorted(os.listdir(img_folder))

        # get length and width
        img_path = os.path.join(self.img_folder, self.img_list[0])
        img = Image.open(img_path)
        self.img_width, self.img_height = img.size
        self.total_img = len(self.img_list)

        print("Total length is ", len(self.img_list) * self.img_width * self.img_height)

    def __len__(self):
        return len(self.img_list) * self.img_width * self.img_height

    def __getitem__(self, index):

        z = index // (self.img_width * self.img_height) # Image (z is actually a time coordinate)
        x = index % self.img_width # x position of an image
        y = (index // self.img_width) % self.img_height # y position of an image

        img_path = os.path.join(self.img_folder, self.img_list[z])
        img = Image.open(img_path).convert('RGB')

        rgb = torch.div(torch.tensor(img.getpixel((x, y))), 255.0)

        sample = {'position': torch.tensor([x, y, z]), 'rgb': rgb}

        return sample
    
class PixelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        super(PixelDataLoader, self).__init__(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)

    def collate_fn(self, batch):
        inputs = [item['position'] for item in batch]
        targets = [item['rgb'] for item in batch]

        inputs_batch = torch.stack(inputs, dim=0)
        targets_batch = torch.stack(targets, dim=0)

        index_batch = torch.arange(len(batch))

        return (inputs_batch, index_batch), (targets_batch, index_batch)

