from typing import Tuple, List

from datasets.loaders import DatasetConfig, SingleSetConfig


class MultiSetConfig(DatasetConfig):
    def __init__(self,
                 tfrecords_config_folders: List[str],
                 output_range: Tuple[float, float]):
        self.tfrecords_config_folders = tfrecords_config_folders
        self.configs = [SingleSetConfig(folder, output_range) for folder in tfrecords_config_folders]

        for config in self.configs:
            config.filter_out_unshared_modalities(self.configs)

        super(MultiSetConfig, self).__init__(modalities=self.configs[0].modalities,
                                             shard_duration=self.configs[0].shard_duration,
                                             video_frequency=self.configs[0].video_frequency,
                                             audio_frequency=self.configs[0].audio_frequency,
                                             statistics=self.configs[0].statistics,
                                             output_range=output_range,
                                             )

    def get_subset_folders(self,
                           subset_name: str
                           ) -> List[str]:
        folders = []
        for config in self.configs:
            folders += config.get_subset_folders(subset_name)

        return folders
