import tensorflow as tf
import numpy as np
import os
import copy
from typing import Dict, Tuple, Optional, List, Generator, Union

from datasets.loaders import DatasetConfig, SingleSetConfig
from datasets.loaders.SubsetLoader import SubsetLoader
from modalities import Modality, ModalityCollection, Pattern, Landmarks, MelSpectrogram
from misc_utils.general import int_ceil, int_floor


class TFRecordSubsetLoader(SubsetLoader):
    # region Initialization
    def __init__(self,
                 config: DatasetConfig,
                 subset_name: str):
        super(TFRecordSubsetLoader, self).__init__(subset_name=subset_name)

        self.config = config
        self.check_unsupported_shard_sizes()

        self.subset_folders = config.get_subset_folders(subset_name)

        self._train_tf_dataset: Optional[tf.data.Dataset] = None
        self._test_tf_dataset: Optional[tf.data.Dataset] = None

    def check_unsupported_shard_sizes(self):
        for modality in self.config.modalities:
            if isinstance(modality, MelSpectrogram):
                # For MelSpectrogram, max_shard_size is always equal to initial_shard_size
                continue

            shard_size = self.config.get_modality_shard_size(modality)
            max_shard_size = int_ceil(shard_size)
            initial_shard_size = int_floor(shard_size)
            if max_shard_size != initial_shard_size:
                raise ValueError("max_shard_size != initial_shard_size : "
                                 "SubsetLoader doesn't support this case yet.")

    # endregion

    # region Make tf.data.Dataset(s)
    def make_tf_dataset(self,
                        pattern: Pattern,
                        seed,
                        subset_folders: List[str] = None,
                        batch_size: int = None,
                        prefetch_size: int = None,
                        parallel_cores=None,
                        ) -> tf.data.Dataset:
        if subset_folders is None:
            subset_folders = self.subset_folders
        subset_folders = copy.copy(subset_folders)

        if isinstance(parallel_cores, str) and parallel_cores == "auto":
            parallel_cores = os.cpu_count()

        shards_per_sample = self.config.compute_shards_per_sample(pattern)

        generator = self.make_shard_filepath_generator(subset_folders, pattern, shards_per_sample, seed=seed)
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=tf.string,
                                                 output_shapes=())
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=parallel_cores)
        dataset = dataset.batch(pattern.modalities_per_sample).prefetch(1)

        dataset = dataset.map(lambda serialized_shards: self.parse_shard(serialized_shards, pattern),
                              num_parallel_calls=parallel_cores)

        dataset = dataset.batch(shards_per_sample)
        dataset = dataset.map(lambda shards, shards_sizes: self.join_shards_and_extract_one_random(shards,
                                                                                                   shards_sizes,
                                                                                                   pattern,
                                                                                                   seed=seed),
                              num_parallel_calls=parallel_cores)

        # dataset = dataset.map(self.normalize_modalities, num_parallel_calls=k)
        # dataset = dataset.map(self.standardize_modalities, num_parallel_calls=k)
        dataset = dataset.map(pattern.apply, num_parallel_calls=parallel_cores)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

            if pattern.batch_processor is not None:
                dataset = dataset.map(pattern.batch_processor, num_parallel_calls=parallel_cores)

        if prefetch_size is not None:
            dataset = dataset.prefetch(prefetch_size)

        return dataset

    def make_tf_datasets_splits(self,
                                pattern: Pattern,
                                split: float,
                                batch_size: int,
                                seed,
                                subset_folders: List[str] = None,
                                parallel_cores=None,
                                split_folders: Dict = None,
                                ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        if (split <= 0.0) or (split >= 1.0):
            raise ValueError("Split must be strictly between 0.0 and 1.0, found {}.".format(split))

        if (split_folders is not None) and (len(split_folders) > 0):
            raise ValueError("When provided, `split_folders` must be an empty dict, which will be filled.")

        if subset_folders is None:
            subset_folders = self.subset_folders
        subset_folders = copy.copy(subset_folders)

        if len(subset_folders) == 1:
            train_dataset = self.make_tf_dataset(pattern, seed=seed,
                                                 subset_folders=subset_folders,
                                                 batch_size=batch_size,
                                                 parallel_cores=parallel_cores)
            validation_dataset = None
            if split_folders is not None:
                split_folders["train_dataset"] = subset_folders
                split_folders["validation_dataset"] = []
            return train_dataset, validation_dataset

        train_count = int_ceil(len(subset_folders) * split)
        np.random.RandomState(seed=seed).shuffle(subset_folders)

        if train_count == len(subset_folders):
            train_count = len(subset_folders) - 1

        train_folders = subset_folders[:train_count]
        validation_folders = subset_folders[train_count:]

        train_dataset = self.make_tf_dataset(pattern, seed=seed, subset_folders=train_folders,
                                             batch_size=batch_size, prefetch_size=-1, parallel_cores=parallel_cores)
        validation_dataset = self.make_tf_dataset(pattern, seed=seed, subset_folders=validation_folders,
                                                  batch_size=batch_size, parallel_cores=parallel_cores)

        print("Train set : {} folders | Validation set : {} folders."
              .format(train_count, len(subset_folders) - train_count))

        if split_folders is not None:
            split_folders["train_dataset"] = train_folders
            split_folders["validation_dataset"] = validation_folders
        return train_dataset, validation_dataset

    def make_source_browser(self,
                            pattern: Pattern,
                            source_index: int,
                            stride: int
                            ) -> Generator:
        source_folder = self.subset_folders[source_index]
        shards_per_sample: int = self.config.compute_shards_per_sample(pattern)

        dataset = self.make_browser_dataset(pattern=pattern, source_folder=source_folder)
        generator = self.browser_generator(dataset=dataset, shards_per_sample=shards_per_sample, pattern=pattern)
        generator = self.strided_generator(generator, stride=stride)

        return generator

    def make_browser_dataset(self,
                             pattern: Pattern,
                             source_folder: str
                             ) -> tf.data.Dataset:

        filepath_generator = self.make_browser_filepath_generator(source_folder, pattern)
        dataset = tf.data.Dataset.from_generator(filepath_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())
        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.batch(pattern.modalities_per_sample).prefetch(1)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, pattern))

        return dataset

    def browser_generator(self,
                          dataset: tf.data.Dataset,
                          shards_per_sample: int,
                          pattern: Pattern):
        modality_ids = pattern.labeled_ids

        unbatched_shards: Dict[str, List[tf.Tensor]] = {modality_id: [] for modality_id in modality_ids}
        unbatched_shards_sizes: Dict[str, List[tf.Tensor]] = {modality_id: [] for modality_id in modality_ids}
        shards: Optional[Dict[str, tf.Tensor]] = None
        shards_sizes: Optional[Dict[str, tf.Tensor]] = None

        for modalities_shard, shard_sizes in dataset:
            stored_shards_count = len(unbatched_shards[modality_ids[0]])

            # region (Re)Fill with next shard(s)
            if stored_shards_count < shards_per_sample:
                for modality_id in unbatched_shards:
                    unbatched_shards[modality_id].append(modalities_shard[modality_id])
                    if modality_id != "labels":
                        unbatched_shards_sizes[modality_id].append(shard_sizes[modality_id])
                stored_shards_count += 1

                if stored_shards_count >= shards_per_sample:
                    shards, shards_sizes = self.batch_shards(unbatched_shards, unbatched_shards_sizes)
                else:
                    continue
            # endregion

            first_shard_size = unbatched_shards_sizes[modality_ids[0]][0].numpy()
            for i in range(first_shard_size):
                offset = i / first_shard_size
                joint_shards = self.join_shards(shards, shards_sizes, offset, pattern)
                joint_shards = pattern.apply(joint_shards)
                yield joint_shards

            for modality_id in unbatched_shards:
                unbatched_shards[modality_id].pop(0)
                if modality_id != "labels":
                    unbatched_shards_sizes[modality_id].pop(0)

        shards, shards_sizes = self.batch_shards(unbatched_shards, unbatched_shards_sizes)
        remaining_length = sum(unbatched_shards_sizes[modality_ids[0]]).numpy()
        sample_length = pattern.flattened[0].length
        remaining_iterations = remaining_length - sample_length

        for i in range(remaining_iterations):
            offset = i / (remaining_iterations - 1)
            joint_shards = self.join_shards(shards, shards_sizes, offset, pattern)
            joint_shards = pattern.apply(joint_shards)
            *data, labels = joint_shards
            joint_shards = (*data, labels)
            yield joint_shards

    # endregion

    # region 1) Generate filepaths
    @staticmethod
    def make_shard_filepath_generator(folders: List[str],
                                      pattern: Pattern,
                                      shards_per_sample: int,
                                      seed):
        modality_ids = list(pattern.modality_ids)
        if pattern.contains_labels:
            modality_ids.append("labels")

        folders = TFRecordSubsetLoader.build_folders_probability_map(folders, modality_ids[0], shards_per_sample)
        generator_random_state = np.random.RandomState(seed=seed)

        def generator():
            # noinspection DuplicatedCode
            while True:
                source_index = generator_random_state.randint(len(folders))
                source_folder = folders[source_index]
                files = []

                shards_count = TFRecordSubsetLoader.get_shards_count(source_folder, modality_ids)
                if shards_count < shards_per_sample:
                    raise RuntimeError("shards_count ({}) < shards_per_sample ({}). "
                                       "You don't have enough samples.".
                                       format(shards_count, shards_per_sample))

                for modality_id in modality_ids:
                    modality_files = TFRecordSubsetLoader.get_modality_files(source_folder, modality_id)
                    files.append(modality_files)

                offset = generator_random_state.randint(shards_count - shards_per_sample + 1)
                for shard_index in range(offset, offset + shards_per_sample):
                    for file_index in range(len(files)):
                        yield files[file_index][shard_index]

        return generator

    @staticmethod
    def build_folders_probability_map(folders: List[str],
                                      modality_ref_id: str,
                                      shards_per_sample: int):
        probability_map = []

        for folder in folders:
            modality_folder = os.path.join(folder, modality_ref_id)
            files = [file for file in os.listdir(modality_folder) if file.endswith(".tfrecord")]
            file_count = len(files)
            ref_count = file_count - (shards_per_sample - 1)
            for _ in range(ref_count):
                probability_map.append(folder)

        return probability_map

    @staticmethod
    def make_browser_filepath_generator(source_folder: str, pattern: Pattern):
        modality_ids = pattern.labeled_ids
        modality_count = len(modality_ids)

        # noinspection DuplicatedCode
        def generator():
            files = []
            shards_count = TFRecordSubsetLoader.get_shards_count(source_folder, modality_ids)

            for modality_id in modality_ids:
                modality_files = TFRecordSubsetLoader.get_modality_files(source_folder, modality_id)
                files.append(modality_files)

            for i in range(shards_count):
                for modality_index in range(modality_count):
                    yield files[modality_index][i]

        return generator

    # endregion

    # region 2) Parse shard
    @tf.function
    def parse_shard(self, serialized_examples, pattern: Pattern):
        serialized_examples.set_shape(pattern.modalities_per_sample)

        features_decoded, modalities_shard_size = {}, {}

        for i, modality_type in enumerate(pattern.required_modality_types):
            modality = self.modalities[modality_type]
            modality_id = modality.id()
            modality_features = modality.tfrecord_features()
            modality_example = serialized_examples[i]

            parsed_features = tf.io.parse_single_example(modality_example, modality_features)

            decoded_modality = modality.decode_from_tfrecord_feature(parsed_features)
            decoded_modality, modality_size = self.pad_modality_if_needed(modality, decoded_modality)

            features_decoded[modality_id] = decoded_modality
            modalities_shard_size[modality_id] = modality_size

        if pattern.contains_labels:
            labels_features = {"labels": tf.io.VarLenFeature(tf.float32)}
            labels_example = serialized_examples[-1]

            parsed_features = tf.io.parse_single_example(labels_example, labels_features)

            labels = parsed_features["labels"].values
            labels = self.pad_labels_if_needed(labels)
            features_decoded["labels"] = labels

        return features_decoded, modalities_shard_size

    @tf.function
    def pad_modality_if_needed(self,
                               modality: Modality,
                               decoded_modality: tf.Tensor
                               ) -> Tuple[tf.Tensor, tf.Tensor]:
        modality_size = tf.shape(decoded_modality)[0]
        modality_max_size = self.config.get_modality_max_shard_size(modality)
        pad_size = modality_max_size - modality_size

        def pad_modality():
            paddings_rank = tf.rank(decoded_modality)
            size_paddings = [[0, pad_size]]
            shape_paddings = tf.zeros(shape=[paddings_rank - 1, 2], dtype=tf.int64)
            paddings = tf.concat([size_paddings, shape_paddings], axis=0,
                                 name=modality.id() + "_paddings")
            return tf.pad(decoded_modality, paddings)

        def identity():
            return decoded_modality

        decoded_modality = tf.cond(pred=pad_size > 0,
                                   true_fn=pad_modality,
                                   false_fn=identity)

        return decoded_modality, modality_size

    @tf.function
    def pad_labels_if_needed(self, labels: tf.Tensor):
        labels_size = tf.shape(labels)[0]
        max_labels_size = self.config.max_labels_size
        pad_size = max_labels_size - labels_size

        def pad_labels():
            paddings = [[pad_size, 0]]
            return tf.pad(labels, paddings)

        def identity():
            return labels

        labels = tf.cond(pred=pad_size > 0,
                         true_fn=pad_labels,
                         false_fn=identity)

        return labels

    # endregion

    # region 3) Join shards
    @tf.function
    def join_shards_and_extract_one_random(self,
                                           shards: Dict[str, tf.Tensor],
                                           shard_sizes: Dict[str, tf.Tensor],
                                           pattern: Pattern,
                                           seed):
        offset = tf.random.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32, name="offset", seed=seed)
        return self.join_shards(shards, shard_sizes, offset, pattern)

    @tf.function
    def join_shards_and_extract_all(self,
                                    shards: Dict[str, tf.Tensor],
                                    shard_sizes: Dict[str, tf.Tensor],
                                    pattern: Pattern,
                                    stride: int
                                    ):
        with tf.name_scope("join_shards_ordered"):
            reference_modality_id = pattern.modality_ids[0]
            size = shard_sizes[reference_modality_id][0]
            stride = tf.constant(stride, tf.int32, name="stride")
            result_count = size // stride

            def loop_body(i, step_shards_arrays: Dict[str, tf.TensorArray]):
                step_offset = tf.cast(i * stride, tf.float32) / tf.cast(size - 1, tf.float32)
                step_joint_shards = self.join_shards(shards, shard_sizes, step_offset, pattern,
                                                     length_map_function=max)
                for modality_id in step_shards_arrays:
                    modality = step_joint_shards[modality_id]
                    step_shards_arrays[modality_id] = step_shards_arrays[modality_id].write(i, modality)
                i += 1
                return i, step_shards_arrays

            i_initializer = tf.constant(0, tf.int32)
            shards_arrays = {modality_id: tf.TensorArray(dtype=shards[modality_id].dtype, size=result_count)
                             for modality_id in shards}
            results = tf.while_loop(cond=lambda i, _: i < result_count,
                                    body=loop_body,
                                    loop_vars=[i_initializer, shards_arrays],
                                    parallel_iterations=1)
            joint_shard: Dict[str, tf.TensorArray] = results[1]
            joint_shard = {modality_id: joint_shard[modality_id].stack(name="stack_{}".format(modality_id))
                           for modality_id in joint_shard}
            return joint_shard

    @tf.function
    def join_shards(self,
                    shards: Dict[str, tf.Tensor],
                    shard_sizes: Dict[str, tf.Tensor],
                    offset: tf.Tensor,
                    pattern: Pattern,
                    length_map_function=max):

        joint_shards = {}
        labels_range = None
        labels_offset = None
        offset = tf.cast(offset, tf.float32)

        # region for each modality : join shards and extract needed length
        for modality in self.modalities:
            modality_type = type(modality)
            if modality_type not in pattern.required_modality_types:
                continue

            modality_id = modality.id()
            modality_shards = shards[modality_id]
            modality_shard_sizes = shard_sizes[modality_id]

            with tf.name_scope(modality_id):
                total_size = tf.cast(tf.reduce_sum(modality_shard_sizes), tf.int32, name="total_shard_size")

                modality_load_infos = pattern.as_dict[modality_type]
                modality_sample_size = length_map_function([load_info.length for load_info in modality_load_infos])
                modality_sample_size_op = tf.constant(modality_sample_size, name="modality_sample_size")

                # region modality_offset = offset * modality_offset_range
                modality_effective_size = modality_shard_sizes[0]
                modality_offset_range = tf.minimum(modality_effective_size, total_size - modality_sample_size_op)
                modality_offset_range = tf.cast(modality_offset_range, tf.float32)
                modality_offset = tf.cast(offset * modality_offset_range, tf.int32, name="offset")
                # endregion

                # region modality_shards = concatenate(modality_shards)
                modality_shards_shape = tf.shape(modality_shards, name="modality_shard_shape")
                modality_shards_shape.set_shape(modality.rank() + 1)
                modality_shards_shape = tf.unstack(modality_shards_shape, name="unstack_modality_shape")
                shards_per_sample, modality_size, *modality_shape = modality_shards_shape
                modality_shards_shape = [shards_per_sample * modality_size, *modality_shape]
                modality_shards = tf.reshape(modality_shards, modality_shards_shape, "concatenate_shards")
                # endregion

                modality_shards = modality_shards[modality_offset:modality_offset + modality_sample_size_op]
                joint_shards[modality_id] = modality_shards

            if "labels" in shards and labels_range is None:
                labels_range = tf.cast(modality_sample_size_op, tf.float32) / tf.cast(total_size, tf.float32)
                size_ratio = tf.cast(modality_effective_size, tf.float32) / tf.cast(total_size, tf.float32)
                labels_offset = offset * size_ratio
        # endregion

        # region if labels in shards: join labels shards and extract needed length
        if "labels" in shards:
            labels: tf.Tensor = shards["labels"]

            shards_per_sample = tf.cast(shards_per_sample, tf.float32)
            shard_labels_offset = tf.range(shards_per_sample, dtype=tf.float32, name="shard_labels_offset")
            shard_labels_offset = tf.expand_dims(shard_labels_offset, axis=-1)

            labels = (labels + shard_labels_offset) / shards_per_sample
            labels = tf.clip_by_value(labels, labels_offset, labels_offset + labels_range)
            labels = (labels - labels_offset) / labels_range

            labels = tf.reshape(labels, shape=[-1, 2])
            joint_shards["labels"] = labels
        # endregion

        return joint_shards

    # endregion

    # region Properties
    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities

    @property
    def source_count(self) -> int:
        return len(self.subset_folders)

    @property
    def size(self) -> int:
        return self.config.get_subset_reference_size(self.subset_name)

    @property
    def sample_names(self) -> List[str]:
        return [os.path.basename(folder) for folder in self.subset_folders]

    @property
    def sample_shape(self) -> Union[List[str], Tuple[str], tf.TensorShape]:
        raise NotImplementedError("The property `sample_shape` must be implemented in subclasses.")

    @property
    def sample_count(self) -> int:
        return len(self.subset_folders)

    # endregion

    # region Utility
    @staticmethod
    def get_shards_count(source_folder: str, modality_ids: List[str]):
        shards_count = None
        for modality_id in modality_ids:
            modality_files = TFRecordSubsetLoader.get_modality_files(source_folder, modality_id)

            if shards_count is None:
                shards_count = len(modality_files)
            elif shards_count != len(modality_files):
                raise ValueError("Modalities don't have the same number of shards in "
                                 "{}.".format(source_folder))

        return shards_count

    @staticmethod
    def get_modality_files(source_folder: str, modality_id: str) -> List[str]:
        modality_folder = os.path.join(source_folder, modality_id)
        modality_files = [os.path.join(modality_folder, file)
                          for file in os.listdir(modality_folder)
                          if file.endswith(".tfrecord")]
        modality_files = list(sorted(modality_files))
        return modality_files

    @staticmethod
    def get_modality_file_count(source_folder: str, modality_id: str) -> int:
        return len(TFRecordSubsetLoader.get_modality_files(source_folder, modality_id))

    def get_sample_tfrecords_count(self, sample_index: int, modality_id="labels"):
        return self.get_modality_file_count(self.subset_folders[sample_index], modality_id)

    @staticmethod
    def batch_shards(unbatched_shards: Dict[str, List[tf.Tensor]],
                     unbatched_shards_sizes: Dict[str, List[tf.Tensor]]
                     ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:

        shards = {}
        shards_sizes = {}
        for modality_id in unbatched_shards:
            shards[modality_id] = tf.stack(unbatched_shards[modality_id], axis=0)
            if modality_id != "labels":
                shards_sizes[modality_id] = tf.stack(unbatched_shards_sizes[modality_id], axis=0)

        return shards, shards_sizes
    # endregion


# region Main (test)
def main():
    from modalities import ModalityLoadInfo, Pattern
    from modalities import RawAudio
    from modalities import MelSpectrogram

    audio_length = int(48000 * 1.28)
    nfft = 52

    pattern = Pattern(
        (
            ModalityLoadInfo(Landmarks, 32),
            ModalityLoadInfo(MelSpectrogram, nfft)
        ),
        (
            ModalityLoadInfo(Landmarks, 128),
            ModalityLoadInfo(RawAudio, audio_length)
        )
    )

    config = SingleSetConfig(tfrecords_config_folder="C:/datasets/emoly_split",
                             output_range=(0.0, 1.0))

    loader = TFRecordSubsetLoader(config, "Test")
    dataset = loader.make_tf_dataset(pattern, seed=0)
    print(dataset)


if __name__ == "__main__":
    main()

# endregion
