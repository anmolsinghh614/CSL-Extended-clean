# imagenet_lt_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

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
        # For validation dataset where images are directly under the root
        if 'ILSVRC2012_img_val' in self.data_root:
            return os.path.join(self.data_root, path)
        
        # Handle cases where validation might still be in subdirectories (unlikely for your case)
        if os.path.isdir(self.data_root):
            return os.path.join(self.data_root, path)
        
        print(f"File not found for path: {path}")
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        for _ in range(len(self.image_paths)): 
            img_path = self.image_paths[index]
            if img_path is None:
                index = (index + 1) % len(self.image_paths)
                continue
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                if self.is_test:
                    return image
                else:
                    label = self.labels[index]
                    return image, label
            
            except FileNotFoundError:
                print(f"File not found: {img_path}")
                index = (index + 1) % len(self.image_paths)
            except Exception as e:
                print(f"Error loading image: {e}")
                index = (index + 1) % len(self.image_paths)  
        
        raise RuntimeError("All files are missing or corrupted.")

    def get_unique_labels(self):
        """Returns the unique labels in the dataset."""
        return set(self.labels) if not self.is_test else set()


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rgb_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0) if key == 'default' else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]) if split == 'train' else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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

def get_imagenet_lt_loaders(train_txt, val_txt, test_txt, train_dir, val_dir, test_dir, batch_size= 256, num_workers=4):
    train_transform = get_data_transform('train', [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_transform = get_data_transform('val', [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = get_data_transform('test', [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset = LT_Dataset(data_root=train_dir, txt_path=train_txt, transform=train_transform)
    val_dataset = LT_Dataset(data_root=val_dir, txt_path=val_txt, transform=val_transform)
    test_dataset = LT_Dataset(data_root=test_dir, txt_path=test_txt, transform=test_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
