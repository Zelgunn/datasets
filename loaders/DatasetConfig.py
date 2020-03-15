import tensorflow as tf
from abc import abstractmethod
import json
import os
from typing import Dict, List, Any, Tuple, Union, Type

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Modality, ModalityCollection, Pattern
from modalities import RawVideo, Faces, OpticalFlow, DoG, Landmarks
from modalities import RawAudio, MelSpectrogram
from misc_utils.general import int_ceil


def get_shard_count(sample_length: int,
                    shard_size: int
                    ) -> int:
    shard_count = 1 + int_ceil((sample_length - 1) / shard_size)
    return max(2, shard_count)


class DatasetConfig(object):
    def __init__(self,
                 modalities: ModalityCollection,
                 shard_duration: float,
                 video_frequency,
                 audio_frequency,
                 statistics: Dict[str, Any],
                 output_range: Tuple[float, float],
                 ):
        self.modalities = modalities
        self.shard_duration = shard_duration
        self.video_frequency = video_frequency
        self.audio_frequency = audio_frequency
        self.statistics = statistics
        self.output_range = output_range

        self.modalities_ranges: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = {}
        for modality_id, modality_statistics in self.modalities_statistics.items():
            modality_min = tf.constant(value=float(modality_statistics["min"]), name="{}_min".format(modality_id))
            modality_max = tf.constant(value=float(modality_statistics["max"]), name="{}_max".format(modality_id))
            self.modalities_ranges[modality_id] = (modality_min, modality_max)

    @abstractmethod
    def get_subset_folders(self,
                           subset_name: str
                           ) -> List[str]:
        raise NotImplementedError

    def get_modality_record_count(self, subset_name: str, modality_id) -> int:
        records_count = 0
        folders = self.get_subset_folders(subset_name)

        for folder in folders:
            folder = os.path.join(folder, modality_id)
            modality_files = [file for file in os.listdir(folder) if file.endswith(".tfrecord")]
            records_count += len(modality_files)

        return records_count

    def get_modality_shard_size(self,
                                modality: Modality
                                ) -> Union[float, int]:

        if isinstance(modality, (RawVideo, Faces, DoG, Landmarks)):
            shard_size = self.video_frequency * self.shard_duration
        elif isinstance(modality, OpticalFlow):
            shard_size = self.video_frequency * self.shard_duration - 1
        elif isinstance(modality, RawAudio):
            shard_size = self.audio_frequency * self.shard_duration
        elif isinstance(modality, MelSpectrogram):
            shard_size = modality.get_output_frame_count(self.shard_duration * self.audio_frequency,
                                                         self.audio_frequency)
        else:
            raise NotImplementedError(modality.id())

        return shard_size

    def get_modality_max_shard_size(self, modality: Modality) -> int:
        return int_ceil(self.get_modality_shard_size(modality))

    def compute_shards_per_sample(self, pattern: Pattern) -> int:
        shard_counts = []
        for modality_load_info in pattern.flattened:
            if isinstance(modality_load_info, str):
                continue

            sample_length = modality_load_info.length
            modality_type = modality_load_info.modality

            modality = self.modalities[modality_type]
            shard_size = self.get_modality_max_shard_size(modality)
            shard_count = get_shard_count(sample_length, shard_size)

            shard_counts.append(shard_count)

        shards_per_sample: int = max(shard_counts)
        return shards_per_sample

    def get_shared_modality_types(self, configs: List["DatasetConfig"]):
        modalities: List[Type[Modality]] = []

        for modality in self.modalities:
            shared = True
            for config in configs:
                if modality.id() not in config.modalities.ids():
                    shared = False
                    break
            if shared:
                modalities.append(type(modality))

        return modalities

    def filter_out_unshared_modalities(self, configs: List["DatasetConfig"]):
        shared_modalities = self.get_shared_modality_types(configs)
        self.modalities.filter(shared_modalities)

        shared_ids = [modality.id() for modality in shared_modalities]
        for modality_id in self.modalities_ids:
            if modality_id not in shared_ids:
                self.modalities_statistics.pop(modality_id)

    @staticmethod
    def load_tf_records_config(tfrecords_config_folder: str) -> Dict[str, Any]:
        tf_records_config_filepath = os.path.join(tfrecords_config_folder, tfrecords_config_filename)
        with open(tf_records_config_filepath, 'r') as file:
            tfrecords_config: Dict[str, Any] = json.load(file)
        return tfrecords_config

    @property
    def modalities_ids(self) -> List[str]:
        return self.modalities.ids()

    @property
    def modalities_statistics(self) -> Dict[str, Dict[str, str]]:
        return self.statistics["modalities"]

    @property
    def max_labels_size(self) -> int:
        return int(self.statistics["max_labels_size"])


def main():
    pass


if __name__ == "__main__":
    main()
