import numpy as np
import os
import argparse
from typing import Tuple, List

from datasets.loaders.nprecords.NumpyDatasetLoader import NumpyDatasetLoader
from datasets.loaders.nprecords.NumpySubsetLoader import NumpySubsetLoader
from datasets.data_readers.packet_readers.CIDDSPacketReader import CIDDSPacketReader, CIDDSNetworkProtocol


class CIDDSLoader(NumpyDatasetLoader):
    def __init__(self,
                 dataset_folder: str,
                 protocol: CIDDSNetworkProtocol,
                 min_normal_segment_length: int,
                 train_samples_ratio=0.8):
        self.protocol = protocol
        self.min_normal_segment_length = min_normal_segment_length
        super(CIDDSLoader, self).__init__(dataset_folder, train_samples_ratio)

    def init_subsets(self):
        normal_segments = get_normal_segments(self.labels, self.min_normal_segment_length)
        train_segments = split_train_segments(normal_segments, ratio=self.train_samples_ratio)
        test_segments = get_test_segments(train_segments, sample_count=self.labels.shape[0])

        train_packets = segment_data(self.packets, train_segments)
        test_packets = segment_data(self.packets, test_segments)

        train_labels = segment_data(self.labels, train_segments)
        test_labels = segment_data(self.labels, test_segments)

        self.subsets["Train"] = NumpySubsetLoader("Train", train_packets, train_labels)
        self.subsets["Test"] = NumpySubsetLoader("Test", test_packets, test_labels)

    def build_packets(self, max_frame_count: int = None) -> np.ndarray:
        packets, _ = self.build_data()
        return packets

    def build_labels(self) -> np.ndarray:
        _, labels = self.build_data()
        return labels

    def build_data(self) -> Tuple[np.ndarray, np.ndarray]:
        reader = CIDDSPacketReader(packet_sources=self.get_csv_filepaths())
        protocols, samples, labels = reader.read_all(stack=True)

        selected_samples, selected_labels = None, None
        for protocol in CIDDSNetworkProtocol:
            rows_matching_protocol = protocols == protocol
            protocol_samples = samples[rows_matching_protocol]
            protocol_labels = labels[rows_matching_protocol]

            protocol_name = (str(protocol)).split('.')[-1]
            protocol_folder = os.path.join(self.dataset_folder, protocol_name)
            if not os.path.exists(protocol_folder):
                os.makedirs(protocol_folder)

            samples_filepath = os.path.join(protocol_folder, "dataset.npy")
            labels_filepath = os.path.join(protocol_folder, "labels.npy")

            np.save(samples_filepath, protocol_samples)
            np.save(labels_filepath, protocol_labels)

            if protocol == self.protocol:
                selected_samples = protocol_samples
                selected_labels = protocol_labels

        return selected_samples, selected_labels

    def get_csv_filepaths(self) -> List[str]:
        files = os.listdir(self.dataset_folder)
        filepaths = [os.path.join(self.dataset_folder, name) for name in files if name.endswith(".csv")]
        return filepaths

    def get_labels_numpy_filepath(self) -> str:
        return os.path.join(self.protocol_folder, "labels.npy")

    def get_packets_numpy_filepath(self) -> str:
        return os.path.join(self.protocol_folder, "dataset.npy")

    @property
    def protocol_name(self) -> str:
        return self.protocol.name

    @property
    def protocol_folder(self) -> str:
        return os.path.join(self.dataset_folder, self.protocol_name)


def get_normal_segments(labels: np.ndarray, min_segment_length: int) -> np.ndarray:
    normal_segments = []
    segment_start = 0
    current_is_anomalous = False

    for i in range(len(labels) - 1):
        current_is_anomalous = labels[i]
        next_is_anomalous = labels[i + 1]

        if current_is_anomalous:
            if not next_is_anomalous:
                segment_length = i - segment_start + 1
                if segment_length > min_segment_length:
                    normal_segments.append((segment_start, i))
        else:
            if next_is_anomalous:
                segment_start = i + 1

    if not current_is_anomalous:
        segment_length = len(labels) - segment_start
        if segment_length > min_segment_length:
            normal_segments.append((segment_start, len(labels) - 1))

    normal_segments = np.asarray(normal_segments, dtype=np.int32)
    return normal_segments


def split_train_segments(normal_segments: np.ndarray,
                         ratio: float
                         ) -> np.ndarray:
    shuffle_indices = np.arange(normal_segments.shape[0])
    np.random.shuffle(shuffle_indices)

    train_total, overall_total = 0, 0
    train_segments = []

    segments_length = normal_segments[:, 1] - normal_segments[:, 0] + 1
    for index in shuffle_indices:
        normal_segment = normal_segments[index]
        segment_length = segments_length[index]

        if train_total == 0:
            current_ratio = 0.0
        else:
            current_ratio = train_total / overall_total

        if current_ratio < ratio:
            train_segments.append(normal_segment)
            train_total += segment_length
        overall_total += segment_length

    train_segments = np.asarray(train_segments)
    sort_indices = np.argsort(train_segments[:, 0])
    train_segments = train_segments[sort_indices]
    return train_segments


def get_test_segments(train_segments: np.ndarray, sample_count: int):
    test_segments = np.reshape(np.copy(train_segments), newshape=[-1])

    if test_segments[0] == 0:
        test_segments = test_segments[1:]
    else:
        test_segments = np.pad(test_segments, pad_width=(1, 0), constant_values=-1)

    if test_segments[-1] == (sample_count - 1):
        test_segments = test_segments[:-1]
    else:
        test_segments = np.pad(test_segments, pad_width=(0, 1), constant_values=sample_count)

    test_segments = np.reshape(test_segments, newshape=[-1, 2])
    offset = np.asarray([+1, -1], dtype=train_segments.dtype)
    test_segments += offset

    return test_segments


def segment_data(data: np.ndarray, segments: np.ndarray) -> List[np.ndarray]:
    segmented_data = []

    for start, end in segments:
        data_segment = data[start: end + 1]
        segmented_data.append(data_segment)

    return segmented_data


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_folder", required=True)
    args = arg_parser.parse_args()
    dataset_folder = args.dataset_folder

    loader = CIDDSLoader(dataset_folder=dataset_folder, protocol=CIDDSNetworkProtocol.TCP, min_normal_segment_length=64)
    loader.build_data()


if __name__ == "__main__":
    main()
