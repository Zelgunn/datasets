import numpy as np
import os
import csv
from typing import Iterator, TextIO, Optional


def get_row_count(filepath) -> int:
    with open(filepath) as file:
        row_count = sum(1 for _ in file)
    return row_count


def get_csv_column_count(filepath) -> int:
    with open(filepath) as file:
        csv_reader = csv.reader(file)
        column_count = len(next(csv_reader))
    return column_count


class PacketReader(object):
    """
    :param packet_source:
        A string (filepath of a csv file).
    :param start:
        (Optional) The start (in number of packets) of the stream. Used to only read a sub-part.
    :param end:
        (Optional) The end (in number of packets) of the stream. Used to only read a sub-part.
    """

    def __init__(self,
                 packet_source: str,
                 discard_first_column=False,
                 start=None,
                 end=None,
                 max_frame_count=None,
                 ):

        if not isinstance(packet_source, str) or not os.path.isfile(packet_source):
            raise ValueError("`packet_source` must be a valid csv file.")

        self.packet_source = packet_source
        self.discard_first_column = discard_first_column
        self.packet_size = get_csv_column_count(packet_source)
        if self.discard_first_column:
            self.packet_size -= 1
        self.file: Optional[TextIO] = None

        # region End / Start
        if max_frame_count is None:
            max_frame_count = get_row_count(packet_source)

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
        elif start < max_frame_count:
            self.start = start
        else:
            raise ValueError("`start` is after the end of the stream "
                             "{}(start) >= {}(frame_count)".format(start, max_frame_count))

        if self.end <= self.start:
            raise ValueError("`start` must be less than `end`, got {}(start) and {}(end)".format(self.start, self.end))
        # endregion

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.file is not None:
            self.close()

        self.file = open(self.packet_source, "r")
        csv_reader = csv.reader(self.file)

        for i, frame in enumerate(csv_reader):
            if i >= self.start:
                if self.discard_first_column:
                    frame = frame[1:]
                frame = np.asarray(frame).astype(np.float32)
                yield frame

            if (i + 1) == self.end:
                break

        self.close()

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
