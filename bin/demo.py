import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=str, required=True)
    main(parser.parse_args())
