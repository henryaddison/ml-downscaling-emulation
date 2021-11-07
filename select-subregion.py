import argparse
import logging
import os
from pathlib import Path

from preprocessing.select_subregion import SelectSubregion

def get_args():
    parser = argparse.ArgumentParser(description='Select a sub-region of a UKCP18 file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Base path to input storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to storage output')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outputs = SelectSubregion(args.input_dir, args.output_dir).run()

    logging.info(outputs)
