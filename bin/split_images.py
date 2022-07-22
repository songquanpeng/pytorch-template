import argparse
import os
import shutil

from tqdm import tqdm


def main(args):
    input_path = args.input_path
    test_num = args.test_num
    img_paths = os.listdir(input_path)
    img_paths.sort()
    train_num = len(img_paths) - test_num
    train_path = f"{input_path}_train_{train_num}"
    test_path = f"{input_path}_test_{test_num}"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        if i < train_num:
            shutil.copy(os.path.join(input_path, img_path), train_path)
        else:
            shutil.copy(os.path.join(input_path, img_path), test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--test_num', type=int, default=10000)
    main(parser.parse_args())
