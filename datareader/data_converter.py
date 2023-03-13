
import sys
sys.path.append("./")
from datareader.fusedchat_reader import FUSEDCHATReader
from datareader.ketod_reader import KETODReader
from datareader.WoI_reader import WOIReader
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Dataset Converter")
    parser.add_argument("--schema_guided", type=str, help="the path of schema guided")

    parser.add_argument("--ketod_dataset", type=str, help="the path of ketod dataset")
    parser.add_argument("--ketod_sample", type=str, help="the path of ketod sample (out file)")

    parser.add_argument("--fusedchat_dataset", type=str, help="the path of ketod dataset")
    parser.add_argument("--fusedchat_sample", type=str, help="the path of ketod sample (out file)")

    parser.add_argument("--woi_dataset", type=str, help="the path of Wizard of internet dataset")
    parser.add_argument("--woi_sample", type=str, help="the path of  Wizard of internet sample (out file)")

    return parser


def ketod_converter(data_path, sample_path, schema_guided):
    ketod = KETODReader(data_path, sample_path, schema_guided)
    ketod.start()
    ketod.__call__()
    ketod.end()


def woi_converter(data_path, sample_path):
    woi = WOIReader(data_path, sample_path)
    woi.start()
    woi.__call__()
    woi.end()


def fusedchat_converter(data_path, sample_path):
    fusedchat = FUSEDCHATReader(data_path, sample_path)
    fusedchat.start()
    fusedchat.__call__()
    fusedchat.end()


if __name__ == '__main__':
    args = make_parser().parse_args()
    ketod_converter(args.ketod_dataset,
                    args.ketod_sample,
                    args.schema_guided)

    woi_converter(args.woi_dataset,
                  args.woi_sample)

    fusedchat_converter(args.fusedchat_dataset,
                        args.fusedchat_sample)
