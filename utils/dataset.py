from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        return image, 0
