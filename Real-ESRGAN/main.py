import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=8)
    model.load_weights('/Users/jaernizam/Desktop/MIPS/Final_Project/Real-ESRGAN/weights/RealESRGAN_x8.pth', download=True)
    image = Image.open("/Users/jaernizam/Desktop/MIPS/Final_Project/Real-ESRGAN/Enhancement/LQ_8x/OAS1-0269_IMG_2320_rotate_-10.041164925232428.png").convert('RGB')
    sr_image = model.predict(image)
    sr_image.save('/Users/jaernizam/Desktop/MIPS/Final_Project/Real-ESRGAN/results/2320_x8.png')

if __name__ == '__main__':
    main()