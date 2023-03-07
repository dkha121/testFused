import sys
sys.path.append("./")

from datareader.ketod_reader import KETODReader
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Dataset Converter")
    parser.add_argument("--ketod_dataset", type=str, help="the path of ketod dataset")
    parser.add_argument("--ketod_sample", type=str, help="the path of ketod sample (out file)")

    return parser


def ketod_converter(data_path, sample_path):
    ketod = KETODReader(data_path, sample_path)
    ketod.__call__()


if __name__ == '__main__':
    args = make_parser().parse_args()
    print("\nStarting to convert KETOD" + "."*10)
    ketod_converter(args.ketod_dataset, args.ketod_sample)
    print("Complete converting KETOD")


