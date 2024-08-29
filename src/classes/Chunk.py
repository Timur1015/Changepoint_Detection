from numpy import ndarray


class Chunk:

    def __init__(self, c_id, data: ndarray):
        """
        Initialize a Chunk instance.

        Args:
            c_id: The unique identifier for the chunk.
            data: The data contained in the chunk, represented as a NumPy ndarray.
        """
        self.c_id = c_id
        self.data = data

    def get_data(self):
        """
        Get the data contained in the chunk.

        Returns:
            The data as a NumPy ndarray.
        """
        return self.data

    def set_data(self, data):
        self.data = data

    def get_id(self):
        """
        Get the unique identifier of the chunk.

        Returns:
             The unique identifier (chunk ID).
        """
        return self.c_id
