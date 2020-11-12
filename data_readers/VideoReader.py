import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
from enum import IntEnum
from typing import Union, List, Optional, Iterator, Tuple

from misc_utils.general import int_floor, int_ceil


class VideoReaderMode(IntEnum):
    CV_VIDEO_CAPTURE = 0,
    NP_ARRAY = 1,
    IMAGE_COLLECTION = 2


class VideoReaderProto(object):
    def __init__(self,
                 video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                 mode: VideoReaderMode = None,
                 frequency=25,
                 start=0,
                 end=None
                 ):
        self.video_source = video_source
        self.mode = mode
        self.frequency = frequency
        self.start = start
        self.end = end

    def to_video_reader(self) -> "VideoReader":
        return VideoReader(video_source=self.video_source,
                           mode=self.mode,
                           frequency=self.frequency,
                           start=self.start,
                           end=self.end)


class VideoReader(object):
    """
        :param video_source:
            Either a string (filepath), a numpy array, a cv2.VideoCapture or an list of string (filepaths to images),
            the source for the video.
        :param mode:
            (Optional) Either `CV_VIDEO_CAPTURE` for audio files on hard drive, or `NP_ARRAY` for numpy arrays
            or `IMAGE_COLLECTION` for a set of images.
        :param frequency:
            (Optional) The video frequency, required for numpy arrays if `start` or `end` are not None.
        :param start:
            (Optional) The start (in seconds) of the audio. Used to only read a sub-part.
        :param end:
            (Optional) The end (in seconds) of the audio. Used to only read a sub-part.
        """

    def __init__(self,
                 video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                 mode: VideoReaderMode = None,
                 frequency=None,
                 start=None,
                 end=None
                 ):

        self.name = None
        if isinstance(video_source, str):
            self.name = video_source

        if mode is None:
            self.mode = infer_video_reader_mode(video_source)
        else:
            assert mode == infer_video_reader_mode(video_source)
            self.mode = mode

        self.video_source = video_source

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_array: Optional[np.array] = None
        self.image_collection: Optional[List[str]] = None

        # region Select & set container
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            if isinstance(video_source, str):
                self.video_capture = cv2.VideoCapture(video_source)

                if not self.video_capture.isOpened():
                    raise RuntimeError("Could not open {}.".format(self.video_source))
            else:
                self.video_capture = video_source

            max_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frequency is None:
                frequency = float(self.video_capture.get(cv2.CAP_PROP_FPS))

        elif self.mode == VideoReaderMode.NP_ARRAY:
            if isinstance(video_source, str):
                self.video_array = np.load(video_source, mmap_mode="r")
            elif isinstance(video_source, np.ndarray):
                self.video_array = video_source
            else:
                self.video_array = np.asarray(video_source)
            max_frame_count = len(self.video_array)

        else:
            if isinstance(video_source, str):
                images_names = os.listdir(video_source)
                self.image_collection = [os.path.join(video_source, image_name) for image_name in images_names
                                         if is_image_format_supported(image_name)]
                self.image_collection = sorted(self.image_collection)
            else:
                self.image_collection = video_source
            max_frame_count = len(self.image_collection)

        self.frequency = frequency
        # endregion

        # region End / Start
        if start is not None:
            start = int(start * frequency)
        if end is not None:
            end = int(end * frequency)

        if end is None:
            self.end = max_frame_count
        elif end < 0:
            self.end = max_frame_count + end
        else:
            self.end = min(end, max_frame_count)

        if start is None:
            self.start = 0
        elif start < 0:
            self.start = max_frame_count + start
        else:
            self.start = start

        if self.end <= self.start:
            raise ValueError("End frame index ({}) is less or equal than"
                             " the start frame index ({}). "
                             "Max frame count is {}. Mode is {}. Frequency is {}. Name is {}."
                             .format(self.end, self.start, max_frame_count, self.mode, self.frequency, self.name))
        # endregion

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start)

        for i in range(self.start, self.end):
            if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
            elif self.mode == VideoReaderMode.NP_ARRAY:
                frame = self.video_array[i]
            else:
                frame = Image.open(self.image_collection[i])
                frame = np.array(frame)

            yield frame

    # region Properties
    @property
    def frame_count(self):
        return self.end - self.start

    @property
    def frame_height(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.height

    @property
    def frame_width(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[2]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.width

    @property
    def frame_channels(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return 3
        elif self.mode == VideoReaderMode.NP_ARRAY:
            if self.video_array.ndim == 4:
                return self.video_array.shape[3]
            else:
                return 1
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            if frame.ndim == 3:
                return frame.shape[2]
            else:
                return 1

    @property
    def frame_shape(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width, 3]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:]
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            return frame.shape

    @property
    def frame_size(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:3]
        else:
            frame = Image.open(self.image_collection[0])
            return tuple(reversed(frame.size))

    # endregion

    def rewrite_video(self, target_path: str, fps: int, frame_size: Tuple[int, int] = None):
        if frame_size is None:
            frame_size = (self.frame_width, self.frame_height)

        out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size)

        for frame in tqdm(self, total=self.frame_count):
            out.write(frame)
        out.release()

    def close(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            self.video_capture.release()

    def read_all(self, frame_size=None) -> np.ndarray:
        if frame_size is None:
            frames = [frame for frame in self]
        else:
            frames = [cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST) for frame in self]
        return np.asarray(frames)

    def extract_most_likely_background(self, buffer_size=None, k=1, frame_size=None):
        frame_size = self.frame_size if frame_size is None else frame_size

        frames = self.read_all()
        frames = tf.convert_to_tensor(frames)

        if frames.shape.rank < 4:
            frames = tf.expand_dims(frames, -1)

        frames = tf.image.resize(frames, frame_size)
        votes_shape = frames.shape[1:] + [256]
        votes = tf.zeros(shape=votes_shape, dtype=tf.int32)

        if frames.dtype != tf.int32:
            if (frames.dtype == tf.float32) and (tf.reduce_all(frames <= 1.0)):
                frames *= 255.0
            frames = tf.cast(frames, tf.int32)

        buffer_size = frames.shape[0] if buffer_size is None else buffer_size
        for j in range(0, frames.shape[0], buffer_size):
            frames_votes = tf.one_hot(frames[j:j + buffer_size], depth=256, dtype=tf.int32)
            frames_votes = tf.reduce_sum(frames_votes, axis=0)
            votes += frames_votes

        k_votes_weights, k_votes = tf.math.top_k(votes, k, sorted=False)
        k_votes_weights, k_votes = tf.cast(k_votes_weights, tf.float32), tf.cast(k_votes, tf.float32)
        k_votes_weights /= tf.reduce_sum(k_votes_weights, axis=-1, keepdims=True)
        background = tf.reduce_sum(k_votes * k_votes_weights, axis=-1) / 255.0

        return background

    def remove_most_likely_background(self, buffer_size=None, frame_size=None, k=1):
        frame_size = self.frame_size if frame_size is None else frame_size

        frames = self.read_all()
        frames = tf.convert_to_tensor(frames)

        if frames.shape.rank < 4:
            frames = tf.expand_dims(frames, -1)

        frames = tf.image.resize(frames, frame_size)
        votes_shape = frames.shape[1:] + [256]
        votes = tf.zeros(shape=votes_shape, dtype=tf.int32)

        if frames.dtype != tf.int32:
            if (frames.dtype == tf.float32) and (tf.reduce_all(frames <= 1.0)):
                frames *= 255.0
            frames = tf.cast(frames, tf.int32)

        buffer_size = frames.shape[0] if buffer_size is None else buffer_size
        for j in range(0, frames.shape[0], buffer_size):
            frames_votes = tf.one_hot(frames[j:j + buffer_size], depth=256, dtype=tf.int32)
            frames_votes = tf.reduce_sum(frames_votes, axis=0)
            votes += frames_votes

        _, k_votes = tf.math.top_k(votes, k, sorted=False)
        k_votes = tf.one_hot(k_votes, depth=256) != 0
        k_votes = tf.reduce_any(k_votes, axis=-2)
        k_votes = tf.expand_dims(k_votes, axis=0)

        frequency_filter = tf.ones(shape=[3, 3, 1, 1], dtype=tf.int32)
        gaussian_filter = get_gaussian_kernel(size=5, std=1.0)
        gaussian_filter = tf.expand_dims(tf.expand_dims(gaussian_filter, axis=-1), axis=-1)

        output_frames = []
        for j in range(0, frames.shape[0], buffer_size):
            buffer = frames[j:j + buffer_size]
            frames_votes = tf.one_hot(buffer, depth=256) != 0
            vote_in_top_k = tf.logical_and(k_votes, frames_votes)
            vote_in_top_k = tf.reduce_any(vote_in_top_k, axis=-1)  # for color depth
            if vote_in_top_k.shape[-1] > 1:
                vote_in_top_k = tf.reduce_any(vote_in_top_k, axis=-1, keepdims=True)  # for rgb

            mask = 1 - tf.cast(vote_in_top_k, tf.int32)
            mask = tf.nn.conv2d(mask, frequency_filter, strides=1, padding="SAME") > 3
            mask = tf.cast(mask, tf.float32)
            mask = tf.nn.conv2d(mask, gaussian_filter, strides=1, padding="SAME")

            masked_frames = tf.cast(buffer, tf.float32) * mask / 255.0
            output_frames.append(masked_frames)

        output_frames = tf.concat(output_frames, axis=0)

        return output_frames


def infer_video_reader_mode(video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]]):
    if isinstance(video_source, cv2.VideoCapture):
        return VideoReaderMode.CV_VIDEO_CAPTURE

    elif isinstance(video_source, np.ndarray):
        assert video_source.ndim == 3 or video_source.ndim == 4
        return VideoReaderMode.NP_ARRAY

    elif isinstance(video_source, list):
        assert all([isinstance(element, str) for element in video_source])
        assert all([os.path.isfile(element) for element in video_source])
        return VideoReaderMode.IMAGE_COLLECTION

    elif not isinstance(video_source, str):
        raise ValueError("\'video_source\' must either be a string, a VideoCapture, a ndarray or a list of strings."
                         "Received `{}` of with type `{}`".format(video_source, type(video_source)))

    elif os.path.isdir(video_source):
        return VideoReaderMode.IMAGE_COLLECTION

    elif os.path.isfile(video_source):
        if ".npy" in video_source or ".npz" in video_source:
            return VideoReaderMode.NP_ARRAY
        else:
            return VideoReaderMode.CV_VIDEO_CAPTURE

    else:
        raise ValueError("\'video_source\' : {} does not exist.".format(video_source))


def is_image_format_supported(image_name: str) -> bool:
    # source : https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html
    image_name = image_name.lower()
    supported_image_formats = [".bmp", ".eps", ".gif", ".icns", ".ico", ".im", ".jpg", ".jpeg", ".msp", ".pcx", ".png",
                               ".ppm", ".sgi", ".spider", ".tif", ".tiff", ".xbm", ".blp", ".cur", ".dcx", ".dds",
                               ".fli", ".flc", ".fpx", ".ftex", ".gbr", ".gd", ".imt", ".iptc", ".naa", ".mcidas",
                               ".mic", ".mpo", ".pcd", ".pixar", ".psd", ".tga", ".wal", ".xpm"]
    for image_format in supported_image_formats:
        if image_name.endswith(image_format):
            return True
    return False


def one_hot_pixels(frame: np.ndarray) -> np.ndarray:
    frame_shape = frame.shape
    indexes = np.reshape(frame, [-1])
    result = np.zeros([frame.size, 256], np.int32)
    result[np.arange(frame.size), indexes] = 1
    result = np.reshape(result, [*frame_shape, 256])
    return result


def get_gaussian_kernel(size: int, std: float) -> tf.Tensor:
    """Makes 2D gaussian Kernel for convolution."""

    distribution = tfp.distributions.Normal(0.0, std)

    left_size = int_floor(size / 2)
    right_size = int_ceil(size / 2)
    values = distribution.prob(tf.range(start=-left_size, limit=right_size, dtype=tf.float32))
    gaussian_kernel = tf.einsum('i,j->ij', values, values)
    gaussian_kernel / tf.reduce_sum(gaussian_kernel)
    gaussian_kernel = tf.constant(gaussian_kernel)
    return gaussian_kernel


def main():
    video_reader = VideoReader(r"..\datasets\ucsd\ped2\Test\Test001_gt")
    print(video_reader.image_collection)
    for frame in video_reader:
        cv2.imshow("frame", frame)
        cv2.waitKey(40)


if __name__ == "__main__":
    main()
