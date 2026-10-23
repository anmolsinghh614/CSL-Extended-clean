import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rgb_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]) if key == 'default' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ])
    }
    return data_transforms[split]

class LT_Dataset(Dataset):
    def __init__(self, data_root, txt_path, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.is_test = False
        
        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    path, label = parts
                    self.image_paths.append(self._construct_image_path(path))
                    self.labels.append(int(label))
                elif len(parts) == 1:
                    path = parts[0]
                    self.image_paths.append(self._construct_image_path(path))
                    self.is_test = True  
                else:
                    raise ValueError(f"Line format incorrect: {line}")

    def _construct_image_path(self, path):
        return os.path.join(self.data_root, path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return self.__getitem__((index + 1) % len(self.image_paths))
        except Exception as e:
            return self.__getitem__((index + 1) % len(self.image_paths))

        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image  
        else:
            label = self.labels[index]
            return image, label

    def get_unique_labels(self):
        if self.is_test:
            raise ValueError("Cannot get unique labels from a test dataset.")
        return set(self.labels)

def get_inaturalist_loaders(train_txt, val_txt, test_txt, train_dir, val_dir, test_dir, batch_size=32, num_workers=4):
    train_transform = get_data_transform('train', [0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    val_transform = get_data_transform('val', [0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    test_transform = get_data_transform('test', [0.466, 0.471, 0.380], [0.195, 0.194, 0.192])

    train_dataset = LT_Dataset(data_root=train_dir, txt_path=train_txt, transform=train_transform)
    val_dataset = LT_Dataset(data_root=val_dir, txt_path=val_txt, transform=val_transform)
    test_dataset = LT_Dataset(data_root=test_dir, txt_path=test_txt, transform=test_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
