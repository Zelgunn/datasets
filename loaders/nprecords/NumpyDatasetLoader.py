from abc import abstractmethod
import numpy as np
import os
from typing import Tuple, Optional

from datasets.loaders.DatasetLoader import DatasetLoader


class NumpyDatasetLoader(DatasetLoader):
    def __init__(self, dataset_folder: str, train_samples_ratio=1.0, ):
        self.dataset_folder = dataset_folder
        self.train_samples_ratio = train_samples_ratio
        self.labels: Optional[np.ndarray] = None
        self.packets: Optional[np.ndarray] = None

        self.load()
        self.normalize_packets()
        super(NumpyDatasetLoader, self).__init__()

    @abstractmethod
    def init_subsets(self):
        raise NotImplementedError("You must implement this method in sub-classes.")

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        self.labels = self.load_labels()
        self.packets = self.load_packets(self.labels.shape[0])
        return self.labels, self.packets

    def load_packets(self, max_frame_count: int = None) -> np.ndarray:
        packets_numpy_filepath = self.get_packets_numpy_filepath()
        if not os.path.exists(packets_numpy_filepath):
            self.print("Could not find packets in `{}`. Building packets...".format(packets_numpy_filepath))
            packets = self.build_packets(max_frame_count=max_frame_count)
            self.print("Packets built. Saving to `{}`.".format(packets_numpy_filepath))
            np.save(packets_numpy_filepath, packets)
        else:
            self.print("Loading packets from `{}`.".format(packets_numpy_filepath))
            packets = np.load(packets_numpy_filepath)
        return packets

    def load_labels(self) -> np.ndarray:
        labels_numpy_filepath = self.get_labels_numpy_filepath()
        if not os.path.exists(labels_numpy_filepath):
            self.print("Could not find labels in `{}`. Building labels...".format(labels_numpy_filepath))
            labels = self.build_labels()
            self.print("Labels built. Saving to `{}`.".format(labels_numpy_filepath))
            np.save(labels_numpy_filepath, labels)
        else:
            self.print("Loading labels from `{}`.".format(labels_numpy_filepath))
            labels = np.load(labels_numpy_filepath)
        return labels

    @abstractmethod
    def build_packets(self, max_frame_count: int = None) -> np.ndarray:
        raise NotImplementedError("You must implement this method in sub-classes.")

    @abstractmethod
    def build_labels(self) -> np.ndarray:
        raise NotImplementedError("You must implement this method in sub-classes.")

    def normalize_packets(self, baseline: np.ndarray = None):
        if baseline is None:
            baseline = self.packets

        baseline_min = np.min(baseline, axis=0, keepdims=True)
        baseline_max = np.max(baseline, axis=0, keepdims=True)
        baseline_range = baseline_max - baseline_min
        if np.any(baseline_range == 0):
            baseline_range = np.where(baseline_range == 0, np.ones_like(baseline_min), baseline_range)

        self.packets = (self.packets - baseline_min) / baseline_range

    def get_labels_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "labels.npy")

    def get_packets_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "dataset.npy")

    def print(self, message: str):
        print("{} - {}".format(self.__class__.__name__, message))
