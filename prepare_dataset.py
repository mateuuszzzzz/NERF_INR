import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_list = sorted(os.listdir(img_folder))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder, self.img_list[index])
        img = Image.open(img_path).convert('RGB')
        return img

class CustomDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        super(CustomDataloader, self).__init__(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)

    def collate_fn(self, batch):
        inputs = []
        targets = []

        for i, img in enumerate(batch):
            for x in range(img.width):
                for y in range(img.height):
                    pixel = img.getpixel((x, y))
                    inputs.append((x, y, i))
                    targets.append(pixel / 255.) # Normalize RGB

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        return inputs, targets
