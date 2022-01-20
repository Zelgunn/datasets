from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Generator

from modalities import Pattern


class SubsetLoader(ABC):
    def __init__(self, subset_name: str):
        self.subset_name = subset_name

    # region Make tf.data.Dataset(s)
    @abstractmethod
    def make_tf_dataset(self,
                        pattern: Pattern,
                        seed,
                        subset_folders: List[str] = None,
                        batch_size: int = None,
                        prefetch_size: int = None,
                        parallel_cores=None,
                        ) -> tf.data.Dataset:
        raise NotImplementedError("The method `make_tf_dataset` must be implemented in subclasses.")

    @abstractmethod
    def make_tf_datasets_splits(self,
                                pattern: Pattern,
                                split: float,
                                batch_size: int,
                                seed,
                                subset_folders: List[str] = None,
                                parallel_cores=None,
                                split_folders: Dict = None,
                                ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        raise NotImplementedError("The method `make_tf_datasets_splits` must be implemented in subclasses.")

    @abstractmethod
    def make_source_browser(self,
                            pattern: Pattern,
                            source_index: int,
                            stride: int
                            ) -> Generator:
        raise NotImplementedError("The method `make_source_browser` must be implemented in subclasses.")

    # endregion

    # region Properties
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError("The property `size` must be implemented in subclasses.")

    @property
    @abstractmethod
    def sample_names(self) -> List[str]:
        raise NotImplementedError("The property `sample_names` must be implemented in subclasses.")

    @property
    @abstractmethod
    def sample_shape(self) -> Union[List[str], Tuple[str], tf.TensorShape]:
        raise NotImplementedError("The property `sample_shape` must be implemented in subclasses.")

    @property
    def sample_count(self) -> int:
        return len(self.sample_names)

    # endregion

    # region Utility

    def get_batch(self, batch_size: int, pattern: Pattern, seed, parallel_cores=None):
        dataset = self.make_tf_dataset(pattern, seed=seed, batch_size=batch_size, parallel_cores=parallel_cores)
        results = None
        for results in dataset:
            break

        return results

    @staticmethod
    def timestamps_labels_to_frame_labels(timestamps: Union[tf.Tensor, np.ndarray],
                                          frame_count: Union[tf.Tensor, int]
                                          ):
        with tf.name_scope("timestamps_labels_to_frame_labels"):
            timestamps = tf.convert_to_tensor(timestamps, dtype=tf.float32)  # shape : [batch_size, pairs_count, 2]
            frame_count = tf.convert_to_tensor(frame_count, dtype=tf.int32)  # shape : []
            frame_count_float = tf.cast(frame_count, tf.float32)

            batch_size, timestamps_per_sample, _ = timestamps.shape
            epsilon = tf.constant(1e-4, dtype=tf.float32, name="epsilon")  # shape : []

            timestamps = tf.expand_dims(timestamps, axis=1)  # shape : [batch_size, 1, pairs_count, 2]
            timestamps = tf.tile(timestamps, multiples=[1, frame_count, 1, 1])
            starts, ends = tf.unstack(timestamps, num=2, axis=-1)  # shape : [batch_size, frame_count, pairs_count] * 2
            delta = tf.abs(ends - starts)  # shape : [batch_size, frame_count, pairs_count]
            labels_are_not_equal = delta > epsilon  # shape : [batch_size, frame_count, pairs_count]

            frame_ids = tf.range(frame_count_float, dtype=tf.float32)  # shape : [frame_count]
            frame_ids = tf.reshape(frame_ids, [1, frame_count, 1])  # shape : [1, frame_count, 1]
            frame_duration = 1.0 / frame_count_float  # shape : []
            start_time = frame_ids * frame_duration  # shape : [1, frame_count, 1]
            end_times = start_time + frame_duration  # shape : [1, frame_count, 1]

            # shape (out) : [batch_size, frame_count, pairs_count]
            start_in = tf.logical_and(start_time >= starts, start_time <= ends)
            end_in = tf.logical_and(end_times >= starts, end_times <= ends)

            frame_in = tf.logical_or(start_in, end_in)  # shape : [batch_size, frame_count, pairs_count]
            frame_in = tf.logical_and(frame_in, labels_are_not_equal)  # shape : [batch_size, frame_count, pairs_count]
            frame_labels = tf.reduce_any(frame_in, axis=-1)  # shape : [batch_size, frame_count]
        return frame_labels

    @staticmethod
    def strided_generator(generator: Generator, stride: int):
        k = 0
        for x in generator:
            if k == 0:
                yield x
            k = (k + 1) % stride

    # endregion
