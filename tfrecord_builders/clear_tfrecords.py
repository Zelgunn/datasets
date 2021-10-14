import os
import argparse

from misc_utils.general import list_dir_recursive


def clear_tfrecords(path):
    for file in list_dir_recursive(path, ".tfrecord"):
        os.remove(file)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_path")
    args = arg_parser.parse_args()
    dataset_path: str = args.dataset_path
    clear_tfrecords(dataset_path)


if __name__ == "__main__":
    main()
