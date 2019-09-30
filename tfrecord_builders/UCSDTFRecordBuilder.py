import os
from typing import Tuple, List, Union, Optional

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.data_readers import VideoReader


class UCSDTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 verbose=1):
        super(UCSDTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                  shard_duration=shard_duration,
                                                  video_frequency=video_frequency,
                                                  audio_frequency=None,
                                                  modalities=modalities,
                                                  labels_frequency=video_frequency,
                                                  video_buffer_frame_size=video_buffer_frame_size,
                                                  verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_sample_count(self, subset: str) -> int:
        subset_folder = os.path.join(self.dataset_path, subset)
        folders = os.listdir(subset_folder)
        folders = [folder for folder in folders
                   if os.path.isdir(os.path.join(subset_folder, folder))
                   and not folder.endswith("_gt")]
        return len(folders)

    def get_data_sources(self) -> List[DataSource]:
        subsets_names = ("Test", "Train")
        subsets_lengths = {subset_name: self.get_sample_count(subset_name) for subset_name in subsets_names}
        subsets = {}
        for subset in subsets_lengths:
            samples = []
            for i in range(subsets_lengths[subset]):
                filename = "{subset}/{subset}{index:03d}".format(subset=subset, index=i + 1)
                path = os.path.join(self.dataset_path, filename)
                path = os.path.normpath(path)

                label_path = "{}_gt".format(path)
                if not os.path.exists(label_path):
                    label_path = False

                sample = (path, label_path)
                samples.append(sample)
            subsets[subset] = samples

        data_sources = [DataSource(labels_source=labels,
                                   target_path=path,
                                   subset_name=subset,
                                   video_source=VideoReader(path),
                                   video_frame_size=self.video_frame_size)
                        for subset in subsets
                        for path, labels in subsets[subset]]
        return data_sources


if __name__ == "__main__":
    ucsd_tf_record_builder = UCSDTFRecordBuilder(dataset_path="../datasets/ucsd/ped1",
                                                 shard_duration=2.0,
                                                 video_frequency=25,
                                                 modalities=ModalityCollection(
                                                     [
                                                         RawVideo(),
                                                     ]
                                                 ),
                                                 video_frame_size=(128, 128),
                                                 video_buffer_frame_size=(128, 128),
                                                 )
    ucsd_tf_record_builder.build()
