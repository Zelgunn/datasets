import tensorflow as tf
import numpy as np
import os
import json
import time
import datetime
from multiprocessing import Pool
from typing import Union, Tuple, List, Dict, Type, Optional, Any

from modalities import Modality, ModalityCollection
from modalities.utils import float_list_feature
from datasets.modality_builders import ModalityBuilder, VideoBuilder, AudioBuilder, BuildersList
from datasets.data_readers import AudioReader
from datasets.data_readers.VideoReader import VideoReaderProto
from datasets.labels_builders import LabelsBuilder
from misc_utils.math_utils import join_two_distributions_statistics


class DataSource(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[float, float]]],
                 target_path: str,
                 subset_name: str,
                 video_source: Union[str, np.ndarray, List[str], VideoReaderProto] = None,
                 video_frame_size: Tuple[int, int] = None,
                 audio_source: Union[AudioReader, str, np.ndarray] = None
                 ):
        self.labels_source = labels_source
        self.target_path = target_path
        self.subset_name = subset_name

        self.video_source = video_source
        self.video_frame_size = video_frame_size
        self.audio_source = audio_source


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
    def __init__(self):
        self.modalities_statistics: Dict[Type[Modality], ModalityShardStatistics] = {}
        self.max_labels_size = 0

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
                 labels_frequency: Union[int, float] = None,
                 video_buffer_frame_size: Tuple[int, int] = None,
                 verbose=1):

        self.dataset_path = dataset_path
        self.shard_duration = shard_duration
        if video_frequency is None and audio_frequency is None:
            raise ValueError("You must specify at least the frequency for either Video or Audio, got None and None")
        self.video_frequency = video_frequency
        self.audio_frequency = audio_frequency
        self.labels_frequency = labels_frequency
        self.modalities = modalities
        self.video_buffer_frame_size = video_buffer_frame_size
        self.verbose = verbose

    def get_data_sources(self) -> List[DataSource]:
        raise NotImplementedError("`get_dataset_sources` should be defined for subclasses.")

    def build(self, core_count=6):
        data_sources = self.get_data_sources()

        subsets_dict: Dict[str, List[str]] = {"Train": [], "Test": []}

        data_sources_count = len(data_sources)
        start_time = time.time()

        builder_pool = Pool(core_count)
        builders = []
        builders_outputs = BuilderOutput()

        for data_source in data_sources:
            # region Fill subsets_dict with folders containing shards
            target_path = os.path.relpath(data_source.target_path, self.dataset_path)
            if data_source.subset_name in subsets_dict:
                subsets_dict[data_source.subset_name].append(target_path)
            else:
                subsets_dict[data_source.subset_name] = [target_path]
            # endregion

            builder = builder_pool.apply_async(self.build_one, (data_source,))
            builders.append(builder)

        working_builders = builders
        while len(working_builders) > 0:
            remaining_builders = []

            for builder in working_builders:
                if builder.ready():
                    builder_output: BuilderOutput = builder.get()
                    builders_outputs.update(builder_output)
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

        tfrecords_config = {
            "modalities": self.modalities.get_config(),
            "shard_duration": self.shard_duration,
            "video_frequency": self.video_frequency,
            "audio_frequency": self.audio_frequency,
            "subsets": subsets_dict,
            "statistics": builders_outputs.to_dict(),
        }

        # TODO : Merge previous tfrecords_config with new when adding new modalities
        with open(os.path.join(self.dataset_path, tfrecords_config_filename), 'w') as file:
            json.dump(tfrecords_config, file)

    def build_one(self, data_source: Union[DataSource, List[DataSource]]) -> BuilderOutput:
        builders = self.make_builders(video_source=data_source.video_source,
                                      video_frame_size=data_source.video_frame_size,
                                      video_buffer_frame_size=self.video_buffer_frame_size,
                                      audio_source=data_source.audio_source)

        modality_builder = BuildersList(builders=builders)

        # TODO : Delete previous .tfrecords
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

        if self.verbose > 0:
            print("\r{} : Done".format(data_source.target_path))

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

        return builders
