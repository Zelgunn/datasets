import tensorflow as tf
import numpy as np
import os
from typing import Dict, Tuple, Optional, List, Generator

from modalities import Pattern, NetworkPacket

from datasets.loaders.SubsetLoader import SubsetLoader


class KitsuneSubsetLoader(SubsetLoader):
    def __init__(self, subset_name: str, packets: np.ndarray, labels: np.ndarray):
        super(KitsuneSubsetLoader, self).__init__(subset_name=subset_name)
        self.packets = tf.convert_to_tensor(packets, dtype=tf.float32)
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
                                         batch_size=1)

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

        indices = np.arange(start=0, stop=self.size - slice_length + 1, step=stride, dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices(indices)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.size, seed=seed)

        slice_extractor = self.make_slice_extractor(length=slice_length, include_labels=pattern.contains_labels)
        dataset = dataset.map(slice_extractor, num_parallel_calls=parallel_cores)
        dataset = dataset.map(pattern.apply, num_parallel_calls=parallel_cores)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        # def tmp_a(x, y):
        #     tf.print(tf.shape(x), tf.shape(y))
        #     return x, y
        #
        # def tmp_b(x):
        #     tf.print(tf.shape(x))
        #     return x
        #
        # if pattern.contains_labels:
        #     dataset = dataset.map(tmp_a)
        # else:
        #     dataset = dataset.map(tmp_b)

        if prefetch_size is not None:
            dataset = dataset.prefetch(prefetch_size)

        return dataset

    def make_slice_extractor(self, length: int, include_labels: bool):
        _length = tf.constant(length, dtype=tf.int32)

        @tf.function
        def extract_packets_slice(index: tf.Tensor) -> Dict[str, tf.Tensor]:
            packets = self.packets[index:index + _length]
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
        return self.packets.shape[0]

    @property
    def sample_names(self) -> List[str]:
        return [self.subset_name]

    @property
    def sample_count(self) -> int:
        return 1

    @property
    def packet_size(self) -> int:
        return self.packets.shape[1]
    # endregion
