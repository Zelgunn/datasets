import os
from typing import Tuple, List, Union, Optional, NamedTuple
from enum import IntEnum

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.data_readers.VideoReader import VideoReaderProto


class SubwayTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 version: "SubwayVideo" = None,
                 verbose=1):
        super(SubwayTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                    shard_duration=shard_duration,
                                                    video_frequency=video_frequency,
                                                    audio_frequency=audio_frequency,
                                                    modalities=modalities,
                                                    video_buffer_frame_size=video_buffer_frame_size,
                                                    verbose=verbose)
        if version is None:
            version = self.guess_version(dataset_path)
            if version is None:
                raise ValueError("Could not determine version from dataset_path {} and version is `None`.".
                                 format(dataset_path))

        self.version = version
        self.video_frame_size = video_frame_size

    @staticmethod
    def guess_version(dataset_path: str) -> Optional["SubwayVideo"]:
        known_alias = {
            "exit": SubwayVideo.EXIT,
            "entrance": SubwayVideo.ENTRANCE,
            "mall1": SubwayVideo.MALL1,
            "mall2": SubwayVideo.MALL2,
            "mall3": SubwayVideo.MALL3,
        }

        for alias in known_alias:
            if alias in dataset_path:
                return known_alias[alias]

        return None

    def get_data_sources(self) -> List[DataSource]:
        video_filepath = os.path.join(self.dataset_path, self.video_config.video_filename)

        # region train_data_source
        train_video_reader = VideoReaderProto(video_filepath, end=self.video_config.training_frames)
        train_labels = False
        train_target_path = os.path.join(self.dataset_path, "Train")
        if not os.path.isdir(train_target_path):
            os.makedirs(train_target_path)
        train_data_source = DataSource(labels_source=train_labels,
                                       target_path=train_target_path,
                                       subset_name="Train",
                                       video_source=train_video_reader,
                                       video_frame_size=self.video_frame_size)
        # endregion

        # region test_data_source
        test_video_reader = VideoReaderProto(video_filepath, start=self.video_config.training_frames)
        test_labels = self.video_config.get_anomaly_timestamps_in_seconds()
        test_target_path = os.path.join(self.dataset_path, "Test")
        if not os.path.isdir(test_target_path):
            os.makedirs(test_target_path)
        test_data_source = DataSource(labels_source=test_labels,
                                      target_path=test_target_path,
                                      subset_name="Test",
                                      video_source=test_video_reader,
                                      video_frame_size=self.video_frame_size)
        # endregion

        data_sources = [train_data_source, test_data_source]
        return data_sources

    @property
    def video_config(self) -> "SubwayVideoConfig":
        return known_subway_configs[self.version]


class SubwayVideo(IntEnum):
    EXIT = 0
    ENTRANCE = 1
    MALL1 = 2
    MALL2 = 3
    MALL3 = 4


class SubwayVideoConfig(NamedTuple):
    video_filename: str
    training_minutes: float
    fps: int
    anomaly_timestamps: List[Tuple[int, int]]

    @property
    def training_frames(self) -> int:
        return int(self.fps * self.training_minutes * 60)

    def get_anomaly_timestamps_in_seconds(self) -> List[Tuple[float, float]]:
        in_seconds = [
            (
                (start - self.training_frames) / self.fps,
                (end - self.training_frames) / self.fps
            )
            for start, end in self.anomaly_timestamps
        ]
        return in_seconds


# region Pre-defined subsets configurations
exit_config = SubwayVideoConfig(
    video_filename="",
    training_minutes=10.0,
    fps=25,
    anomaly_timestamps=[(40880, 41160), (41400, 41700), (50410, 50710), (50980, 51250), (60160, 60940)]
)

entrance_config = SubwayVideoConfig(
    video_filename="subway_entrance_turnstiles.AVI",
    training_minutes=18.0,
    fps=25,
    anomaly_timestamps=[
        (27900, 28000), (29750, 29850), (39465, 39565), (67700, 67900), (69240, 69340), (69700, 70000), (72095, 72165),
        (73025, 73075), (73750, 74050), (83415, 83485), (84315, 84400), (85780, 85880), (86475, 86540), (88500, 88640),
        (89720, 89800), (95285, 95385), (96715, 96755),
        (100200, 100425), (115470, 115525), (115800, 115970), (116200, 116225), (117580, 117610), (117760, 117900),
        (118235, 118270), (118700, 119100), (119285, 119300), (124700, 124850), (128025, 128100), (130480, 130675),
    ]
)

mall1_config = SubwayVideoConfig(
    video_filename="mall_1.AVI",
    training_minutes=20.0,
    fps=25,
    anomaly_timestamps=[
        (68700, 68800), (69320, 69440), (70035, 70200), (70400, 70515), (70650, 70760), (71350, 71460), (71910, 72000),
        (72450, 72550), (74020, 74110), (74410, 74500), (75935, 76010), (76650, 76750), (77225, 77300), (77750, 77830),
        (78280, 78370), (78825, 78920), (79205, 79300), (79585, 79700), (80625, 80700), (81250, 81320), (82140, 82235)
    ]
)

mall2_config = SubwayVideoConfig(
    video_filename="mall_2.AVI",
    training_minutes=20.0,
    fps=25,
    anomaly_timestamps=[
        # (6040, 6170),
        (33155, 33230), (44600, 44560), (47570, 44660), (48020, 48130), (48640, 48740), (49090, 49175), (49225, 49300),
        (50090, 50190), (50840, 50915), (51220, 51300), (51605, 51680), (51940, 52020), (52310, 52415), (52740, 52815),
        (53145, 53200), (53770, 53850)
    ]
)

mall3_config = SubwayVideoConfig(
    video_filename="mall_3.AVI",
    training_minutes=9.0,
    fps=25,
    anomaly_timestamps=[
        (14000, 14030), (18280, 18350), (29095, 29135), (34290, 34425), (34750, 34820), (35275, 35400), (35730, 35790),
        (36340, 36420), (36940, 37000), (37780, 37850), (38200, 38260), (38680, 38715), (38950, 39000), (39250, 39310),
        (39610, 39650), (41210, 41250), (41775, 41850), (42120, 42160), (42470, 42520), (61515, 61570), (69075, 69135)
    ]
)

known_subway_configs = {
    SubwayVideo.EXIT: exit_config,
    SubwayVideo.ENTRANCE: entrance_config,
    SubwayVideo.MALL1: mall1_config,
    SubwayVideo.MALL2: mall2_config,
    SubwayVideo.MALL3: mall3_config,
}
# endregion


if __name__ == "__main__":
    subway_tf_record_builder = SubwayTFRecordBuilder(dataset_path="../datasets/subway/entrance",
                                                     shard_duration=1.28,
                                                     video_frequency=25,
                                                     audio_frequency=None,
                                                     modalities=ModalityCollection(
                                                         [
                                                             RawVideo(),
                                                         ]
                                                     ),
                                                     video_frame_size=(256, 256),
                                                     video_buffer_frame_size=(256, 256)
                                                     )
    subway_tf_record_builder.build()
