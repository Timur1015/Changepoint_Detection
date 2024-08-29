import concurrent.futures
from src.classes.Chunk import Chunk
from src.classes.CPDetector import CPDetector


class ChunkProcessor:
    def __init__(self, chunks: list, cpd_method: CPDetector, num_workers):
        """
        Initialize the ChunkProcessor.

        Args:
            chunks: List of Chunk objects to be processed.
            cpd_method: An instance of CPDetector used for detecting change points.
            num_workers: Number of worker processes to use for parallel processing.
        """
        self.chunks = chunks
        self.cpd_method = cpd_method
        self.num_workers = num_workers
        self.results = []
        self.total_cps = self.cpd_method.get_n_cps()
        self._adjust_number_of_cps(self.total_cps)

    def process_chunks(self):
        """
        Process all chunks using the CPDetector method.
        """
        self._process_all_chunks()
        self.cpd_method.set_n_cps(self.total_cps)

    def _process_chunk(self, chunk: Chunk):
        """
        Process a single chunk.

        Args:
            chunk: The Chunk object to be processed.

        Returns:
            The processed Chunk object.
        """
        print("Processing chunk_nr: " + str(chunk.get_id()) + "...")
        chunk.set_data(self.cpd_method.run(chunk.get_data()))
        return chunk

    def _process_all_chunks(self):
        """
        Process all chunks in parallel using ProcessPoolExecutor.
        """
        chunks_copy = list(self.chunks)
        print('Queue size of Chunks: ' + str(len(self.chunks)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_chunk, chunk): chunk for chunk in
                       chunks_copy[:min(self.num_workers, len(chunks_copy))]}
            for future in concurrent.futures.as_completed(futures):
                chunk = futures[future]
                result = future.result()
                self.chunks.remove(chunk)
                self.results.append(result)
                print(
                    "Chunk_nr: " + str(chunk.get_id()) + " has processed successfully. Number of changpoints found: " +
                    str(len(result.get_data()) - 1))
        if not self.chunks:
            return
        else:
            self.process_chunks()

    def get_results(self):
        """
        Get the results of the processed chunks.

        Returns:
            List of processed Chunk objects.
        """
        return self.results

    def set_chunks(self, chunks: list[Chunk]):
        """
        Set the chunks to be processed.

        Args:
            chunks: List of Chunk objects to be processed.
        """
        self.chunks = chunks

    def _adjust_number_of_cps(self, old_n_cps):
        """
        Adjust the number of change points to be detected per chunk.

        Args:
            old_n_cps: The original number of change points to be detected.
        """
        n_chunks = len(self.chunks)
        new_n_cps = (old_n_cps / n_chunks) + 2  # +2 to consider irregularities
        print('adjusted number of cps per chunk: ' + str(new_n_cps))
        self.cpd_method.set_n_cps(new_n_cps)
