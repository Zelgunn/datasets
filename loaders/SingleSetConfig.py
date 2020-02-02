import os
from typing import List, Tuple

from datasets.loaders import DatasetConfig
from modalities import ModalityCollection


class SingleSetConfig(DatasetConfig):
    def __init__(self,
                 tfrecords_config_folder: str,
                 output_range: Tuple[float, float],
                 ):
        self.tfrecords_config_folder = tfrecords_config_folder
        self.tfrecords_config = DatasetConfig.load_tf_records_config(tfrecords_config_folder)
        self.subsets = self.tfrecords_config["subsets"]

        modalities = ModalityCollection.from_config(self.tfrecords_config["modalities"])
        super(SingleSetConfig, self).__init__(modalities=modalities,
                                              shard_duration=float(self.tfrecords_config["shard_duration"]),
                                              video_frequency=self.tfrecords_config["video_frequency"],
                                              audio_frequency=self.tfrecords_config["audio_frequency"],
                                              max_labels_size=int(self.tfrecords_config["max_labels_size"]),
                                              modalities_ranges=self.tfrecords_config["modalities_ranges"],
                                              output_range=output_range,
                                              )

    def get_subset_folders(self,
                           subset_name: str
                           ) -> List[str]:
        subset = self.subsets[subset_name]
        folders = []
        for folder in subset:
            folder = os.path.join(self.tfrecords_config_folder, folder)
            folder = os.path.normpath(folder)
            folders.append(folder)
        return folders
