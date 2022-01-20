import numpy as np
import tensorflow as tf
import os
from typing import List, Dict, Tuple, Optional, Union

from modalities import Pattern, NetworkPacket
from datasets.loaders import SubsetLoader


class NumpySubsetLoader(SubsetLoader):
    def __init__(self,
                 subset_name: str,
                 samples: Union[np.ndarray, List[np.ndarray]],
                 labels: Union[np.ndarray, List[np.ndarray]],
                 ):
        super(NumpySubsetLoader, self).__init__(subset_name=subset_name)
        self.multi_array = isinstance(samples, list)
        if self.multi_array:
            segment_lengths = []
            for segment in samples:
                segment_lengths.append(segment.shape[0])

            samples = np.concatenate(samples, axis=0)
            labels = np.concatenate(labels, axis=0)
            self._segment_lengths = np.asarray(segment_lengths, dtype=np.int32)
        else:
            self._segment_lengths = None

        self.samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    def make_tf_dataset(self,
                        pattern: Pattern,
                        seed,
                        subset_folders: List[str] = None,
                        batch_size: int = None,
                        prefetch_size: int = None,
                        parallel_cores=None,
                        ) -> tf.data.Dataset:
        dataset = self.make_base_tf_dataset(pattern=pattern,
                                            stride=1,
                                            seed=seed,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            prefetch_size=prefetch_size,
                                            parallel_cores=parallel_cores)
        dataset = dataset.repeat(-1)
        return dataset

    def make_tf_datasets_splits(self,
                                pattern: Pattern,
                                split: float,
                                batch_size: int,
                                seed,
                                subset_folders: List[str] = None,
                                parallel_cores=None,
                                split_folders: Dict = None
                                ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        train_dataset = self.make_tf_dataset(pattern=pattern,
                                             batch_size=batch_size,
                                             seed=seed,
                                             parallel_cores=parallel_cores)
        return train_dataset, None

    def make_source_browser(self,
                            pattern: Pattern,
                            source_index: int,
                            stride: int
                            ):
        return self.make_base_tf_dataset(pattern=pattern,
                                         stride=stride,
                                         seed=None,
                                         shuffle=False,
                                         batch_size=None)

    def make_base_tf_dataset(self,
                             pattern: Pattern,
                             stride: int,
                             seed,
                             shuffle: bool,
                             batch_size: int = None,
                             prefetch_size: int = None,
                             parallel_cores=None,
                             ) -> tf.data.Dataset:
        if isinstance(parallel_cores, str) and parallel_cores == "auto":
            parallel_cores = os.cpu_count()

        slice_length = pattern.flattened[0].length
        dataset = self.make_indices_tf_dataset(slice_length, stride, shuffle, seed)

        slice_extractor = self.make_slice_extractor(length=slice_length, include_labels=pattern.contains_labels)
        dataset = dataset.map(slice_extractor, num_parallel_calls=parallel_cores)
        dataset = dataset.map(pattern.apply, num_parallel_calls=parallel_cores)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        if prefetch_size is not None:
            dataset = dataset.prefetch(prefetch_size)

        return dataset

    def make_indices_tf_dataset(self, slice_length: int, stride: int, shuffle: bool, seed: int) -> tf.data.Dataset:
        if self.multi_array:
            indices = []
            i = 0
            for segment_length in self._segment_lengths:
                start = i
                end = i + segment_length - slice_length
                segment_indices = np.arange(start=start, stop=end, step=stride, dtype=np.int32)
                indices.append(segment_indices)
                i += segment_length
            indices = np.concatenate(indices, axis=0)
        else:
            indices = np.arange(start=0, stop=self.size - slice_length + 1, step=stride, dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices(indices)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.size, seed=seed)
        return dataset

    def make_slice_extractor(self, length: int, include_labels: bool):
        _length = tf.constant(length, dtype=tf.int32)

        @tf.function
        def extract_packets_slice(index: tf.Tensor) -> Dict[str, tf.Tensor]:
            packets = self.samples[index:index + _length]
            packets.set_shape([length, self.packet_size])
            return {NetworkPacket.id(): packets}

        @tf.function
        def extract_packets_and_labels_slice(index: tf.Tensor) -> Dict[str, tf.Tensor]:
            packets = extract_packets_slice(index)
            labels = self.labels[index:index + _length]
            labels.set_shape([length])
            return {**packets, "labels": labels}

        if include_labels:
            return extract_packets_and_labels_slice
        else:
            return extract_packets_slice

    # region Properties
    @property
    def size(self) -> int:
        return self.samples.shape[0]

    @property
    def sample_names(self) -> List[str]:
        return [self.subset_name]

    @property
    def sample_count(self) -> int:
        return 1

    @property
    def sample_shape(self) -> Union[List[str], Tuple[str], tf.TensorShape]:
        return self.samples.shape[1:]

    @property
    def packet_size(self) -> int:
        return self.samples.shape[1]
    # endregion
