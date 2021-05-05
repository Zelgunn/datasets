import numpy as np
import os
import csv

from datasets.loaders.DatasetLoader import DatasetLoader
from datasets.loaders.kitsune.KitsuneSubsetLoader import KitsuneSubsetLoader
from datasets.data_readers import PacketReader


class KitsuneLoader(DatasetLoader):
    def __init__(self, dataset_folder: str, is_mirai: bool, train_samples_ratio=1.0, ):
        self.dataset_folder = dataset_folder
        self.is_mirai = is_mirai
        self.train_samples_ratio = train_samples_ratio
        self.labels = self.load_labels()
        self.packets = self.load_packets(self.labels.shape[0])
        self.normalize_packets()
        super(KitsuneLoader, self).__init__()

    def init_subsets(self):
        train_packets = self.packets[:self.train_samples_count]
        test_packets = self.packets[self.benign_samples_count:]

        train_labels = self.labels[:self.train_samples_count]
        test_labels = self.labels[self.benign_samples_count:]

        self.subsets["Train"] = KitsuneSubsetLoader("Train", train_packets, train_labels)
        self.subsets["Test"] = KitsuneSubsetLoader("Test", test_packets, test_labels)
        if self.train_samples_ratio < 1.0:
            benign_packets = self.packets[self.train_samples_count:self.benign_samples_count]
            benign_labels = self.labels[self.train_samples_count:self.benign_samples_count]
            self.subsets["Benign"] = KitsuneSubsetLoader("Benign", benign_packets, benign_labels)

    def get_labels_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("labels.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_labels_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "labels.npy")

    def get_packets_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("dataset.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_packets_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "dataset.npy")

    def load_labels(self) -> np.ndarray:
        if not os.path.exists(self.get_labels_numpy_filepath()):
            labels_filepath = self.get_labels_csv_filepath()
            with open(labels_filepath, "r") as file:
                csv_reader = csv.reader(file)
                if not self.is_mirai:
                    next(csv_reader)  # skip header
                labels = [row[-1] == "1" for row in csv_reader]
                labels = np.asarray(labels).astype(np.float32)
            np.save(self.get_labels_numpy_filepath(), labels)
        else:
            labels = np.load(self.get_labels_numpy_filepath())
        return labels

    def load_packets(self, max_frame_count: int = None) -> np.ndarray:
        if not os.path.exists(self.get_packets_numpy_filepath()):
            packets_file = self.get_packets_csv_filepath()
            packets_reader = PacketReader(packets_file, max_frame_count=max_frame_count,
                                          discard_first_column=self.is_mirai)
            packets = np.stack([packet for packet in packets_reader], axis=0)
            np.save(self.get_packets_numpy_filepath(), packets)
        else:
            packets = np.load(self.get_packets_numpy_filepath())
        return packets

    def normalize_packets(self):
        train_packets = self.packets[:self.benign_samples_count]

        train_min = train_packets.min(axis=0, keepdims=True)
        train_max = train_packets.max(axis=0, keepdims=True)
        train_range = train_max - train_min
        if np.any(train_range == 0):
            train_range = np.where(train_range == 0, np.ones_like(train_min), train_range)

        self.packets = (self.packets - train_min) / train_range

    @property
    def benign_samples_count(self) -> int:
        return 1000000 if not self.is_mirai else 100000

    @property
    def train_samples_count(self) -> int:
        return int(self.train_samples_ratio * self.benign_samples_count)
