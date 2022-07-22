import argparse
import os

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def resize(input_path, output_path, target_size=256):
    os.makedirs(output_path, exist_ok=True)
    img_paths = os.listdir(input_path)
    for img_path in tqdm(img_paths):
        img = Image.open(os.path.join(input_path, img_path))
        img = transforms.Resize(target_size)(img.convert('RGB'))
        img.save(os.path.join(output_path, img_path))


def main(args):
    resize(args.input_path, args.output_path, args.target_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--target_size', type=int, default=256)
    main(parser.parse_args())
