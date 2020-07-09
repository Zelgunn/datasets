import os
import csv
from tqdm import tqdm
import argparse
import subprocess
import pytube
from pytube.exceptions import VideoUnavailable, RegexMatchError
from urllib.error import HTTPError, URLError
from typing import Dict, Tuple, List, Union, Optional

from modalities import ModalityCollection
from datasets.tfrecord_builders import TFRecordBuilder, DataSource


class AudioSetTFRB(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 verbose=1):
        super(AudioSetTFRB, self).__init__(dataset_path=dataset_path,
                                           shard_duration=shard_duration,
                                           video_frequency=video_frequency,
                                           audio_frequency=audio_frequency,
                                           modalities=modalities,
                                           video_frame_size=video_frame_size,
                                           video_buffer_frame_size=video_buffer_frame_size,
                                           verbose=verbose)

    def get_data_sources(self) -> List[DataSource]:
        videos_folder = os.path.join(self.dataset_path, "videos")
        video_filenames = os.listdir(videos_folder)

        subset_name = "Train"
        labels = [(0.0, 0.0)]

        data_sources: List[DataSource] = []
        for video_filename in video_filenames:
            sample_name = video_filename[:-4]
            target_path = os.path.join(self.dataset_path, subset_name, sample_name)
            video_path = os.path.join(videos_folder, video_filename)

            data_source = DataSource(labels_source=labels,
                                     target_path=target_path,
                                     subset_name=subset_name,
                                     video_source=video_path,
                                     video_frame_size=self.video_frame_size,
                                     audio_source=video_path
                                     )
            data_sources.append(data_source)

        return data_sources

    @staticmethod
    def prepare_dataset(dataset_path: str, filters: List[str]):
        samples = AudioSetSample.get_samples(dataset_path, filters, min_duration=4.0)
        total_duration = int(sum([sample.duration for sample in samples]) / 3600)
        print("Trying to download {} samples, for a total duration of {} hours.".format(len(samples), total_duration))

        videos_path = os.path.join(dataset_path, "videos")
        if not os.path.exists(videos_path):
            os.makedirs(videos_path)

        skip_list_path = os.path.join(dataset_path, "skip_list.txt")
        skip_list = read_skip_list(skip_list_path)

        success_count = 0
        total_count = 0
        for sample in tqdm(samples):
            if sample.youtube_id in skip_list:
                success = False
            else:
                success = sample.download(videos_path)
                if not success:
                    skip_list.append(sample.youtube_id)
                    write_skip_list(skip_list_path, skip_list)

            if success:
                success_count += 1
            total_count += 1

            success_rate = round(success_count / total_count * 100, ndigits=1)
            if success:
                print("Successfully downloaded video `{}` - success rate : {}%".format(sample.youtube_id, success_rate))
            else:
                print("Failed to get video `{}` - success rate : {}%".format(sample.youtube_id, success_rate))


# YTID, start_seconds, end_seconds, positive_labels
# --4gqARaEJE, 0.000, 10.000, "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"

class AudioSetSample(object):
    def __init__(self, youtube_id: str, start_seconds: float, end_seconds: float, labels: List[int]):
        self.youtube_id = youtube_id
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.labels = labels

    def download(self, folder_path: str):
        output_filepath = os.path.join(folder_path, self.filename)
        if os.path.exists(output_filepath):
            return True

        success = download_youtube_video(self.youtube_id, output_filepath=output_filepath,
                                         start_seconds=self.start_seconds, end_seconds=self.end_seconds)

        return success

    @property
    def filename(self) -> str:
        return self.youtube_id.replace("-", "_") + ".mp4"

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    @staticmethod
    def get_labels_map(dataset_path: str) -> Dict[str, str]:
        labels_map = {}

        map_path = os.path.join(dataset_path, "class_labels_indices.csv")
        with open(map_path, "r") as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # skip header
            for row in reader:
                index, label_id, label_name = row
                labels_map[label_name] = label_id

        return labels_map

    @staticmethod
    def get_samples(dataset_path: str, filters: List[str], min_duration: float) -> List["AudioSetSample"]:
        sources = ["balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv"]
        sources = [os.path.join(dataset_path, source) for source in sources]

        labels_map = AudioSetSample.get_labels_map(dataset_path)
        filters = [labels_map[_filter] for _filter in filters]

        samples = []
        for source in sources:
            with open(source, "r") as file:
                reader = csv.reader(file, delimiter=',')
                for _ in range(3):
                    next(reader)

                for row in reader:
                    youtube_id, start_seconds, end_seconds, *sample_labels = row
                    sample_labels[0] = sample_labels[0][2:]
                    sample_labels[-1] = sample_labels[-1][:-1]
                    start_seconds, end_seconds = float(start_seconds), float(end_seconds)

                    is_long_enough = (end_seconds - start_seconds) > min_duration

                    if len(filters) > 0:
                        has_matching_label = any([(sample_label in filters) for sample_label in sample_labels])
                    else:
                        has_matching_label = True

                    if is_long_enough and has_matching_label:
                        sample = AudioSetSample(youtube_id, start_seconds, end_seconds, sample_labels)
                        samples.append(sample)

        return samples


def download_youtube_video(youtube_id: str,
                           output_filepath: str,
                           start_seconds: float,
                           end_seconds: float
                           ) -> bool:
    # if not video_available(youtube_id):
    #     return False

    urls = get_youtube_download_urls(youtube_id)

    if urls is not None:
        video_url, audio_url = urls
        start = to_timestamp(start_seconds)
        duration = to_timestamp(end_seconds - start_seconds)
        download_command = "ffmpeg -hide_banner -loglevel 0 " \
                           "-ss {start} -i {video_url} " \
                           "-ss {start} -i {audio_url} " \
                           "-t {duration} " \
                           "-map 0:v -map 1:a -c:v libx264 -c:a aac " \
                           "{output_filepath}".format(start=start,
                                                      duration=duration,
                                                      video_url=video_url,
                                                      audio_url=audio_url,
                                                      output_filepath=output_filepath)
        subprocess.run(download_command)
        return True
    else:
        return False


def get_youtube_download_urls(youtube_id: str) -> Optional[Tuple[str, str]]:
    get_url_command = "youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -g https://www.youtube.com/watch?v={}". \
        format(youtube_id)
    command_result = subprocess.run(get_url_command, stdout=subprocess.PIPE)
    command_result = command_result.stdout.decode("utf-8").split("\n")

    if len(command_result) == 1:
        return None

    if len(command_result) == 2:
        # video_url, _ = command_result
        # urls = (video_url, video_url)
        return None
    else:
        video_url, audio_url, _ = command_result
        urls = (video_url, audio_url)
    return urls


def video_available(youtube_id: str) -> bool:
    available = True
    try:
        pytube.YouTube(url="https://www.youtube.com/watch?v={}".format(youtube_id))
    except (VideoUnavailable, ValueError, RegexMatchError, KeyError, HTTPError, URLError):
        available = False
    return available


def to_timestamp(t: float) -> str:
    minutes = int(t / 60)
    seconds = int(t) - minutes * 60
    return "{:02d}:{:02d}".format(minutes, seconds)


def read_skip_list(filepath: str) -> List[str]:
    skip_list = []

    if not os.path.exists(filepath):
        return skip_list

    with open(filepath, "r") as file:
        for line in file:
            if "\n" in line:
                line = line.replace("\n", "")

            if len(line) > 0:
                skip_list.append(line)

    return skip_list


def write_skip_list(filepath: str, skip_list: List[str]):
    with open(filepath, "w") as file:
        for element in skip_list:
            file.write(element + "\n")


def main():
    dataset_path = r"..\datasets\audioset"
    # filters = [
    #     "Speech",
    #     "Male speech, man speaking",
    #     "Female speech, woman speaking",
    #     "Child speech, kid speaking",
    #     "Narration, monologue",
    # ]

    # AudioSetTFRB.prepare_dataset(dataset_path, filters)

    from modalities import RawVideo
    # from modalities import Faces
    # from modalities import OpticalFlow
    # from modalities import DoG
    from modalities import RawAudio
    from modalities import MelSpectrogram
    # from modalities import Landmarks

    parser = argparse.ArgumentParser()
    parser.add_argument("--core_count", default=8, type=int)
    args = parser.parse_args()

    tf_record_builder = AudioSetTFRB(dataset_path=dataset_path,
                                     shard_duration=1.28,
                                     video_frequency=25,
                                     audio_frequency=48000,
                                     modalities=ModalityCollection(
                                         [
                                             RawVideo(),
                                             RawAudio(),
                                             MelSpectrogram(window_width=0.03,
                                                            window_step=0.01005,
                                                            mel_filters_count=128,
                                                            to_db=True)
                                         ]
                                     ),
                                     video_frame_size=(128, 128),
                                     # video_buffer_frame_size=(1080 // 4, 1920 // 4),  # for Faces/Landmarks
                                     video_buffer_frame_size=(128, 128),
                                     )
    tf_record_builder.build(core_count=args.core_count)


if __name__ == "__main__":
    main()
