import torch
from models.generator import Generator
import argparse
from config import NUM_IMAGES, IMAGE_SIZE

def generate_images(num_images):
    generator = Generator()
    generator.load_state_dict(torch.load('models/gan_model.pth'))
    generator.eval()

    for i in range(num_images):
        noise = torch.randn(1, 100)
        fake_image = generator(noise)
        # Save the generated image here...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images using a trained GAN.')
    parser.add_argument('--num_images', type=int, default=NUM_IMAGES, help='Number of images to generate')
    args = parser.parse_args()
    generate_images(args.num_images)
