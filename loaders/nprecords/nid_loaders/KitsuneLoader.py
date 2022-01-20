import numpy as np
import os
import csv

from datasets.loaders.nprecords.NumpyDatasetLoader import NumpyDatasetLoader
from datasets.loaders.nprecords.NumpySubsetLoader import NumpySubsetLoader
from datasets.data_readers import KitsunePacketReader


class KitsuneLoader(NumpyDatasetLoader):
    def __init__(self, dataset_folder: str, is_mirai: bool, train_samples_ratio=1.0, ):
        self.is_mirai = is_mirai
        super(KitsuneLoader, self).__init__(dataset_folder, train_samples_ratio)

    def init_subsets(self):
        train_packets = self.packets[:self.train_samples_count]
        test_packets = self.packets[self.benign_samples_count:]

        train_labels = self.labels[:self.train_samples_count]
        test_labels = self.labels[self.benign_samples_count:]

        self.subsets["Train"] = NumpySubsetLoader("Train", train_packets, train_labels)
        self.subsets["Test"] = NumpySubsetLoader("Test", test_packets, test_labels)
        if self.train_samples_ratio < 1.0:
            benign_packets = self.packets[self.train_samples_count:self.benign_samples_count]
            benign_labels = self.labels[self.train_samples_count:self.benign_samples_count]
            self.subsets["Benign"] = NumpySubsetLoader("Benign", benign_packets, benign_labels)

    def get_labels_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("labels.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_packets_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("dataset.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def build_packets(self, max_frame_count: int = None) -> np.ndarray:
        packets_file = self.get_packets_csv_filepath()
        packets_reader = KitsunePacketReader(packets_file, max_frame_count=max_frame_count, is_mirai=self.is_mirai)
        packets = packets_reader.read_all(stack=True)
        return packets

    def build_labels(self) -> np.ndarray:
        labels_filepath = self.get_labels_csv_filepath()
        with open(labels_filepath, "r") as file:
            csv_reader = csv.reader(file)
            if not self.is_mirai:
                next(csv_reader)  # skip header
            labels = [row[-1] == "1" for row in csv_reader]
            labels = np.asarray(labels).astype(np.float32)
        return labels

    def normalize_packets(self, baseline: np.ndarray = None):
        if baseline is None:
            baseline = self.packets[:self.benign_samples_count]
        super(KitsuneLoader, self).normalize_packets(baseline)

    @property
    def benign_samples_count(self) -> int:
        return 1000000 if not self.is_mirai else 100000

    @property
    def train_samples_count(self) -> int:
        return int(self.train_samples_ratio * self.benign_samples_count)
