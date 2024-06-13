import torch
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from models.generator import Generator
from utils.dataset import ImageDataset
from config import EPOCHS, BATCH_SIZE, LR

def train():
    dataset = ImageDataset('data/images')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    criterion = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for real_images, _ in dataloader:
            # Training steps here...
            pass

if __name__ == "__main__":
    train()
