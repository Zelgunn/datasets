import numpy as np
import os
import csv
from typing import Iterator, TextIO, Optional, List, Union


def get_csv_column_count(filepath: Union[str, List[str]]) -> int:
    if isinstance(filepath, (list, tuple)):
        filepath = filepath[0]

    with open(filepath) as file:
        csv_reader = csv.reader(file)
        column_count = len(next(csv_reader))
    return column_count


class CSVWrapper(object):
    """
    :param sources:
        A string (filepath of a csv file) or a list of strings.
    """

    def __init__(self, sources: Union[str, List[str]]):
        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        for source in sources:
            if not isinstance(source, str) or not os.path.isfile(source):
                raise ValueError("`sources` must either be a valid csv file or a list of valid csv files.")

        self.sources = sources
        self.column_count = get_csv_column_count(sources)
        self._row_count = None
        self.file: Optional[TextIO] = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.file is not None:
            self.close()

        for packet_source in self.sources:
            self.file = open(packet_source, "r")
            csv_reader = csv.reader(self.file)

            for frame in csv_reader:
                yield frame

        self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def get_row_count(self) -> int:
        if self._row_count is None:
            row_count = 0
            for source in self.sources:
                with open(source) as file:
                    row_count += sum(1 for _ in file)
            self._row_count = row_count

        return self._row_count
