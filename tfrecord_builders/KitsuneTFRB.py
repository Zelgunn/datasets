import numpy as np
import os
import csv
import argparse
from tqdm import tqdm
from typing import List, Optional
from enum import IntEnum

from modalities import ModalityCollection, NetworkPacket
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.modality_builders.PacketBuilder import DEFAULT_PACKET_FREQUENCY
from datasets.data_readers import PacketReader


class KitsuneDataset(IntEnum):
    ACTIVE_WIRETAP = 0
    ARP_MITM = 1
    FUZZING = 2
    MIRAI_BOTNET = 3
    OS_SCAN = 4
    SSDP_FLOOD = 5
    SSL_RENEGOTIATION = 6
    SYN_DOS = 7
    VIDEO_INJECTION = 8


class KitsuneTFRB(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 modalities: ModalityCollection,
                 version: KitsuneDataset = None,
                 verbose=1):
        super(KitsuneTFRB, self).__init__(dataset_path=dataset_path,
                                          shard_duration=shard_duration,
                                          video_frequency=None,
                                          audio_frequency=None,
                                          modalities=modalities,
                                          labels_frequency=DEFAULT_PACKET_FREQUENCY,
                                          verbose=verbose)

        if version is None:
            version = self.guess_version(dataset_path)
            if version is None:
                raise ValueError("Could not determine version from dataset_path {} and version is `None`.".
                                 format(dataset_path))

        self.version = version

    @staticmethod
    def guess_version(dataset_path: str) -> Optional[KitsuneDataset]:
        known_alias = {
            "Active Wiretap": KitsuneDataset.ACTIVE_WIRETAP,
            "ARP MitM": KitsuneDataset.ARP_MITM,
            "Fuzzing": KitsuneDataset.FUZZING,
            "Mirai Botnet": KitsuneDataset.MIRAI_BOTNET,
            "OS Scan": KitsuneDataset.OS_SCAN,
            "SSDP Flood": KitsuneDataset.SSDP_FLOOD,
            "SSL Renegotiation": KitsuneDataset.SSL_RENEGOTIATION,
            "SYN DoS": KitsuneDataset.SYN_DOS,
            "Video Injection": KitsuneDataset.VIDEO_INJECTION,
        }

        for alias in known_alias:
            if alias in dataset_path:
                return known_alias[alias]

        return None

    def get_data_sources(self) -> List[DataSource]:
        labels = self.get_labels()
        train_packet_count = self.get_training_packet_count()

        train_target_path = os.path.join(self.dataset_path, "Train")
        train_packet_source = PacketReader(packet_source=self.get_dataset_csv_filepath(),
                                           discard_first_column=self.discard_first_dataset_column(),
                                           end=train_packet_count)

        train_labels = labels[:train_packet_count]
        train_source = DataSource(labels_source=train_labels,
                                  target_path=train_target_path,
                                  subset_name="Train",
                                  packet_source=train_packet_source)

        test_target_path = os.path.join(self.dataset_path, "Test")
        test_packet_source = PacketReader(packet_source=self.get_dataset_csv_filepath(),
                                          discard_first_column=self.discard_first_dataset_column(),
                                          start=train_packet_count)
        test_labels = labels[train_packet_count:]
        test_source = DataSource(labels_source=test_labels,
                                 target_path=test_target_path,
                                 subset_name="Test",
                                 packet_source=test_packet_source)

        self.prepare_build_metadata(train_packet_source, test_packet_source)
        return [train_source, test_source]

    def prepare_build_metadata(self, train_packet_reader: PacketReader, test_packet_reader: PacketReader):
        packets_mins = None
        packets_maxs = None

        if self.verbose > 0:
            print("KitsuneTFRB - Preparing build metadata : computing min/max from training set")

        for packet in tqdm(train_packet_reader, total=train_packet_reader.frame_count):
            if packets_mins is None:
                packets_mins = packet
                packets_maxs = packet
            else:
                packets_mins = np.min([packets_mins, packet], axis=0)
                packets_maxs = np.max([packets_maxs, packet], axis=0)

        packets_mins = np.expand_dims(packets_mins, axis=0)
        packets_maxs = np.expand_dims(packets_maxs, axis=0)

        self.add_build_metadata(train_packet_reader, "packets_mins", packets_mins)
        self.add_build_metadata(train_packet_reader, "packets_maxs", packets_maxs)
        self.add_build_metadata(test_packet_reader, "packets_mins", packets_mins)
        self.add_build_metadata(test_packet_reader, "packets_maxs", packets_maxs)

    def discard_first_dataset_column(self) -> bool:
        return self.version == KitsuneDataset.MIRAI_BOTNET

    def get_training_packet_count(self) -> int:
        # if self.version == KitsuneDataset.MIRAI_BOTNET or self.version == KitsuneDataset.FUZZING:
        if self.version == KitsuneDataset.MIRAI_BOTNET:
            return 55000
        else:
            # return 500000
            return 1000000

    def get_dataset_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_path)
        filename = [name for name in files if name.endswith("dataset.csv")][0]
        filepath = os.path.join(self.dataset_path, filename)
        return filepath

    def get_labels_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_path)
        filename = [name for name in files if name.endswith("labels.csv")][0]
        filepath = os.path.join(self.dataset_path, filename)
        return filepath

    def get_labels(self) -> np.ndarray:
        if self.verbose > 0:
            print("KitsuneTFRB - Getting labels...")

        filepath = self.get_labels_csv_filepath()
        with open(filepath, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header
            labels = [row[-1] == "1" for row in csv_reader]
            labels = np.asarray(labels).astype(np.float32)
        return labels


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_path", nargs="+")
    arg_parser.add_argument("--core_count", default=2, type=int)
    arg_parser.add_argument("--shard_duration", default=16.0, type=float)

    args = arg_parser.parse_args()
    dataset_paths: List[str] = args.dataset_path
    core_count: int = args.core_count
    shard_duration: float = args.shard_duration

    for dataset_path in dataset_paths:
        tf_record_builder = KitsuneTFRB(dataset_path=dataset_path,
                                        shard_duration=shard_duration,
                                        modalities=ModalityCollection([NetworkPacket()])
                                        )
        tf_record_builder.build(core_count=core_count)


if __name__ == "__main__":
    main()
