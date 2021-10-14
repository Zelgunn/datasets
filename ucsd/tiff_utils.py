import os
import argparse
from skimage import io as skimage_io


def update_tiffs(source_path: str, target_path: str):
    sub_directories = [dir_info[0] for dir_info in os.walk(source_path)]
    images_path = []
    for sub_dir in sub_directories:
        files = os.listdir(sub_dir)
        for file in files:
            if file.endswith(".tif"):
                images_path += [os.path.join(sub_dir, file)]

    for image_path in images_path:
        target_image_path = image_path.replace(source_path, target_path)
        dir_name = os.path.dirname(target_image_path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        # noinspection PyUnresolvedReferences
        image = skimage_io.imread(image_path)
        # noinspection PyUnresolvedReferences
        skimage_io.imsave(target_image_path, image)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source_path")
    arg_parser.add_argument("--target_path")

    args = arg_parser.parse_args()
    source_path: str = args.source_path
    target_path: str = args.target_path

    update_tiffs(source_path, target_path)


if __name__ == "__main__":
    main()