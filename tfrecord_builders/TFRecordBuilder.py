import tensorflow as tf
import numpy as np
import os
import json
import time
import datetime
import shutil
from multiprocessing import Pool
from typing import Union, Tuple, List, Dict, Type, Optional, Any, Iterable

from modalities import Modality, ModalityCollection
from modalities.utils import float_list_feature
from datasets.modality_builders import ModalityBuilder, VideoBuilder, AudioBuilder, BuildersList, PacketBuilder
from datasets.data_readers import AudioReader, PacketReader
from datasets.data_readers.VideoReader import VideoReaderProto
from datasets.labels_builders import LabelsBuilder
from misc_utils.math_utils import join_two_distributions_statistics
from misc_utils.general import list_dir_recursive


class DataSource(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[float, float]]],
                 target_path: str,
                 subset_name: str,
                 video_source: Union[str, np.ndarray, List[str], VideoReaderProto] = None,
                 video_frame_size: Tuple[int, int] = None,
                 audio_source: Union[AudioReader, str, np.ndarray] = None,
                 packet_source: Union[PacketReader, str] = None,
                 ):
        self.labels_source = labels_source
        self.target_path = target_path
        self.subset_name = subset_name

        self.video_source = video_source
        self.video_frame_size = video_frame_size
        self.audio_source = audio_source
        self.packet_source = packet_source


class ModalityShardStatistics(object):
    def __init__(self,
                 min_value: float,
                 max_value: float,
                 mean: float,
                 stddev: float,
                 size: int,
                 ):
        self.min_value = min_value
        self.max_value = max_value
        self.mean = mean
        self.stddev = stddev
        self.size = size

    def update(self, other: "ModalityShardStatistics"):
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)

        self.size, self.mean, self.stddev = join_two_distributions_statistics(count_1=self.size, count_2=other.size,
                                                                              mean_1=self.mean, mean_2=other.mean,
                                                                              stddev_1=self.stddev,
                                                                              stddev_2=other.stddev)

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "min": float(self.min_value),
            "max": float(self.max_value),
            "mean": float(self.mean),
            "stddev": float(self.stddev),
            "size": int(self.size),
        }


class BuilderOutput(object):
    def __init__(self,
                 modalities_statistics: Dict[Type[Modality], ModalityShardStatistics] = None,
                 max_labels_size=0):
        if modalities_statistics is None:
            modalities_statistics = {}
        self.modalities_statistics = modalities_statistics
        self.max_labels_size = max_labels_size

    def shard_update(self,
                     shard_statistics: Dict[Type[Modality], ModalityShardStatistics],
                     max_labels_size: int):
        for modality_type, modality_statistics in shard_statistics.items():
            if modality_type not in self.modalities_statistics:
                self.modalities_statistics[modality_type] = modality_statistics
            else:
                self.modalities_statistics[modality_type].update(modality_statistics)

        self.max_labels_size = max(self.max_labels_size, max_labels_size)

    def update(self, other: "BuilderOutput"):
        self.shard_update(shard_statistics=other.modalities_statistics, max_labels_size=other.max_labels_size)

    def to_dict(self) -> Dict[str, Any]:
        final_statistics = {"max_labels_size": self.max_labels_size}

        modalities_statistics = {}
        for modality_type, modality_statistics in self.modalities_statistics.items():
            modalities_statistics[modality_type.id()] = modality_statistics.to_dict()

        final_statistics["modalities"] = modalities_statistics

        return final_statistics


tfrecords_config_filename = "tfrecords_config.json"


class TFRecordBuilder(object):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int] = None,
                 labels_frequency: Union[int, float] = None,
                 video_buffer_frame_size: Tuple[int, int] = None,
                 verbose=1):

        self.dataset_path = dataset_path
        self.shard_duration = shard_duration
        self.video_frequency = video_frequency
        self.audio_frequency = audio_frequency
        self.labels_frequency = labels_frequency
        self.modalities = modalities
        self.video_buffer_frame_size = video_buffer_frame_size
        self.video_frame_size = video_frame_size
        self.verbose = verbose

        self.build_metadata: Dict[Iterable, Dict[str, Any]] = {}

    def get_data_sources(self) -> List[DataSource]:
        raise NotImplementedError("`get_dataset_sources` should be defined for subclasses.")

    def clear_tfrecords(self):
        for file in list_dir_recursive(self.dataset_path, ".tfrecord"):
            os.remove(file)

    def build(self, core_count=6):
        self.clear_tfrecords()
        data_sources = self.get_data_sources()

        data_sources_count = len(data_sources)
        start_time = time.time()

        builder_pool = Pool(core_count)
        builders = []
        builders_infos: Dict[Any, Tuple[str, str]] = {}
        builders_outputs: Dict[Any, BuilderOutput] = {}

        for data_source in data_sources:
            builder = builder_pool.apply_async(self.build_one, (data_source,))
            builders.append(builder)

            target_path = os.path.relpath(data_source.target_path, self.dataset_path)
            builders_infos[builder] = (data_source.subset_name, target_path)

        working_builders = builders
        while len(working_builders) > 0:
            remaining_builders = []

            for builder in working_builders:
                if builder.ready():
                    builder_output: BuilderOutput = builder.get()
                    builders_outputs[builder] = builder_output
                else:
                    remaining_builders.append(builder)

            # region Print ETA
            if self.verbose > 0 and len(working_builders) != len(remaining_builders):
                i = len(builders) - len(remaining_builders)
                elapsed_time = time.time() - start_time
                eta = elapsed_time * (data_sources_count / i - 1)
                eta = datetime.timedelta(seconds=np.round(eta))
                print("Building {}/{} - ETA: {}".format(i, data_sources_count, eta))
            # endregion

            time.sleep(10.0)

            working_builders = remaining_builders

        # region Fill subsets_dict with folders containing shards
        subsets_dict: Dict[str, List[str]] = {}
        statistics: Dict[str, Dict[str, Dict[str, Any]]] = {}  # value = statistics[subset][source][stat_name]

        for builder in builders:
            subset_name, target_path = builders_infos[builder]
            builder_output = builders_outputs[builder]

            if subset_name not in subsets_dict:
                subsets_dict[subset_name] = []
                statistics[subset_name] = {}

            subsets_dict[subset_name].append(target_path)
            statistics[subset_name][target_path] = builder_output.to_dict()
        # endregion

        tfrecords_config = {
            **self.get_config(),
            "subsets": subsets_dict,
            "statistics": statistics,
        }

        with open(os.path.join(self.dataset_path, tfrecords_config_filename), 'w') as file:
            json.dump(tfrecords_config, file)

    def build_one(self, data_source: Union[DataSource, List[DataSource]]) -> BuilderOutput:
        builders = self.make_builders(video_source=data_source.video_source,
                                      video_frame_size=data_source.video_frame_size,
                                      video_buffer_frame_size=self.video_buffer_frame_size,
                                      audio_source=data_source.audio_source,
                                      packet_source=data_source.packet_source)

        modality_builder = BuildersList(builders=builders)

        shard_count = modality_builder.get_shard_count()
        labels_iterator = LabelsBuilder(data_source.labels_source,
                                        shard_count=shard_count,
                                        shard_duration=self.shard_duration,
                                        frequency=self.labels_frequency)

        source_iterator = zip(modality_builder, labels_iterator)

        output = BuilderOutput()

        for i, shard in enumerate(source_iterator):
            # region Verbose
            # if self.verbose > 0:
            #     print("\r{} : {}/{}".format(data_source.target_path, i + 1, shard_count), end='')
            # sys.stdout.flush()
            # endregion

            modalities, labels = shard
            modalities: Dict[Type[Modality], np.ndarray] = modalities
            shard_statistics: Dict[Type[Modality], ModalityShardStatistics] = {}

            for modality_type, modality_value in modalities.items():
                encoded_features = modality_type.encode_to_tfrecord_feature(modality_value)
                self.write_features_to_tfrecord(encoded_features, data_source.target_path, i, modality_type.id())

                # noinspection PyArgumentList
                shard_statistics[modality_type] = ModalityShardStatistics(
                    min_value=modality_value.min(),
                    max_value=modality_value.max(),
                    mean=modality_value.mean(),
                    stddev=modality_value.std(),
                    size=modality_value.shape[0],
                )

            output.shard_update(shard_statistics=shard_statistics, max_labels_size=len(labels))

            features = {"labels": float_list_feature(labels)}
            self.write_features_to_tfrecord(features, data_source.target_path, i, "labels")

        modality_builder.close()

        return output

    @staticmethod
    def write_features_to_tfrecord(features: Dict, base_filepath: str, index: int, sub_folder: str = None):
        example = tf.train.Example(features=tf.train.Features(feature=features))
        if sub_folder is not None:
            base_filepath = os.path.join(base_filepath, sub_folder)
        if not os.path.exists(base_filepath):
            os.makedirs(base_filepath)
        filepath = os.path.join(base_filepath, "shard_{:05d}.tfrecord".format(index))
        writer = tf.io.TFRecordWriter(filepath)
        writer.write(example.SerializeToString())

    def make_builders(self,
                      video_source: Union[str, np.ndarray, List[str], VideoReaderProto],
                      video_frame_size: Tuple[int, int],
                      audio_source: Union[AudioReader, str, np.ndarray],
                      video_buffer_frame_size: Tuple[int, int],
                      packet_source: Union[PacketReader, str],
                      ):

        builders: List[ModalityBuilder] = []

        if VideoBuilder.supports_any(self.modalities):
            video_builder = VideoBuilder(shard_duration=self.shard_duration,
                                         source_frequency=self.video_frequency,
                                         modalities=self.modalities,
                                         video_reader=video_source,
                                         default_frame_size=video_frame_size,
                                         buffer_frame_size=video_buffer_frame_size)
            builders.append(video_builder)

        if AudioBuilder.supports_any(self.modalities):
            audio_builder = AudioBuilder(shard_duration=self.shard_duration,
                                         source_frequency=self.audio_frequency,
                                         modalities=self.modalities,
                                         audio_reader=audio_source)
            builders.append(audio_builder)

        if PacketBuilder.supports_any(self.modalities):
            packets_mins = self.get_build_metadata(packet_source, "packets_mins")
            packets_maxs = self.get_build_metadata(packet_source, "packets_maxs")
            packet_builder = PacketBuilder(self.shard_duration,
                                           modalities=self.modalities,
                                           packet_reader=packet_source,
                                           packets_mins=packets_mins,
                                           packets_maxs=packets_maxs)
            builders.append(packet_builder)

        return builders

    def get_build_metadata(self, reader: Iterable, key: str) -> Optional[Any]:
        if reader in self.build_metadata:
            if key in self.build_metadata[reader]:
                return self.build_metadata[reader][key]
        return None

    def add_build_metadata(self, reader: Iterable, key: str, value: Any):
        if reader not in self.build_metadata:
            self.build_metadata[reader] = {}

        if key in self.build_metadata[reader]:
            if value is not self.build_metadata[reader][key]:
                raise ValueError("Key {} already in `build_metadata` for source {}.".format(key, reader))
        else:
            self.build_metadata[reader][key] = value

    def get_config(self) -> Dict[str, Any]:
        return {
            "shard_duration": self.shard_duration,
            "video_frequency": self.video_frequency,
            "audio_frequency": self.audio_frequency,
            "modalities": self.modalities.get_config(),
            "video_frame_size": self.video_frame_size,
            "video_buffer_frame_size": self.video_buffer_frame_size,
            "labels_frequency": self.labels_frequency,
        }


def copy_tree_no_tfrecord(source, destination):
    if os.path.isdir(source):
        shutil.copytree(source, destination, ignore=ignore_tfrecords)
    else:
        shutil.copy2(source, destination)


def ignore_tfrecords(_, names):
    return [name for name in names if name.endswith(".tfrecord")]
