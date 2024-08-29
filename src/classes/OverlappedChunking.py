from typing import List

from ressources.exceptions.SegmentationError import SegmentationError
from src.classes.Chunk import Chunk


class OverlappedChunking:
    """
       This class implements the overlapped chunking technique for changepoint detection.

       The method is based on the technique described in  P. Fearnhead S. O. Tickle I. A. Eckley und K. Haynes.
       „Parallelization of a Common Changepoint Detection Method“. In: Journal of Computational and Graphical
       Statistics 29.1 (2020), S. 149–161. doi: 10 . 1080 / 10618600 . 2019 . 1647216. eprint: https : / / doi . org
       / 10 . 1080 / 10618600 . 2019 . 1647216. url: https : //doi.org/10.1080/10618600.2019.1647216.

    """
    def __init__(self):
        """chunk data into parts that can be processed with multiprocessing but at least 1 """
        self.n = None
        self.subset_size = None
        self.real_subset_size = None
        self.overlap_region = None
        self.n_chunks = None
        self.chunks = []
        self.inkrement = 0

    def chunk_data(self, dataset, chunk_size, overlap_region, chunk_nr_start):
        """
        Chunk the data into overlapping subsets.

        Args:
            dataset: The dataset to be chunked.
            chunk_size: The size of each chunk.
            overlap_region: The size of the overlap region between chunks.
            chunk_nr_start: The starting chunk number.

        Returns:
            List of created chunks.
        """
        self.overlap_region = overlap_region
        self.n = len(dataset)
        self.subset_size = chunk_size - 2 * overlap_region
        self.n_chunks = self.n // self.subset_size

        rest = self.n % self.subset_size
        actual_chunk_nr = chunk_nr_start
        last_index_reached = False

        # Ensure the overlap_region is appropriate for the given chunk_size (chunk_size - 2 * overlap_region > 0).
        if self.subset_size == 0:
            raise SegmentationError('overlap_region is too large for the given chunk_size. Consider : chunksize - 2 * '
                                    'overlap_region > 0')

        # split the dataset into subsets
        start = 0
        for i in range(self.n_chunks):
            end = start + self.subset_size
            # first and last only needs an overlap_region in one direction
            if i == 0:
                end += overlap_region
            if i > 0:
                start = start - overlap_region
            if i < self.n_chunks - 1:
                end += overlap_region
            # if the end of data set is reached then no more chunking is necessary
            if i == self.n_chunks - 1:
                if rest != 0:
                    end += overlap_region
            if end >= self.n:
                last_index_reached = True
                if rest != 0:
                    end += overlap_region
                self._set_chunk(actual_chunk_nr, dataset, self.calculate_interval_borders(start, end))
                actual_chunk_nr += 1
                break

            self._set_chunk(actual_chunk_nr, dataset, self.calculate_interval_borders(start, end))
            actual_chunk_nr += 1
            start = end - overlap_region

        # Additional chunk for the remaining data
        if rest > 0 and not last_index_reached:
            start = start - overlap_region
            self.chunks.append(Chunk(actual_chunk_nr, dataset[start:self.n]))

        return self.chunks

    # create a new chunk that represents a subset of the general dataset
    def _set_chunk(self, chunk_nr, dataset, interval):
        """
        Create a new chunk that represents a subset of the general dataset.

        Args:
            chunk_nr: The chunk number.
            dataset: The original dataset.
            interval: The interval (start, end) of the chunk within the dataset.
        """
        start, end = interval
        self.chunks.append(Chunk(chunk_nr, dataset[start:end]))

    # the indexes + overlap_region cannot be out of the bounds
    def calculate_interval_borders(self, start, end):
        """
        Calculate the interval borders ensuring they are within dataset bounds.

        Args:
            start: The start index of the interval.
            end: The end index of the interval.

        Returns:
            List containing the adjusted start and end indices.
        """
        return [max(start, 0), min(end, self.n)]

    def merge_chunks(self, chunks: List[Chunk]):
        """
        Merge the chunks back into a single dataset.

        Args:
            chunks: List of chunks to be merged.

        Returns:
            Set of merged results.
        """
        result = []
        merged_result = set()
        sorted_chunks = sorted(chunks, key=lambda x: x.get_id())
        indexes_first_chunk = sorted_chunks[0].get_data()
        for i in range(len(indexes_first_chunk)):
            if i == len(indexes_first_chunk) - 1:
                break
            result.append(indexes_first_chunk[i])
        self.inkrement += indexes_first_chunk[len(indexes_first_chunk) - 1]
        for i in range(1, len(sorted_chunks)):
            adjusted_data = self._adjust_chunk_data(sorted_chunks[i].get_data())
            result.extend(adjusted_data)
            #result.append(adjusted_data)
        for cps in result:
            merged_result.add(cps)
        return merged_result

    # calculate the right position of indexes in general dataset
    def _adjust_chunk_data(self, data: List):
        """
        Adjust the chunk data to its correct position in the general dataset.

        Args:
            data: List of data points in the chunk.

        Returns:
             List of adjusted data points.
        """
        real_p = []
        for i, v in enumerate(data):
            val = v + self.inkrement - 2 * self.overlap_region
            if i == len(data) - 1:
                self.inkrement = val
                continue
            real_p.append(val)
        return real_p

    def reset_inkrement(self):
        """Reset the increment value to zero."""
        self.inkrement = 0
