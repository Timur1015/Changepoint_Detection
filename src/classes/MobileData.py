from enum import Enum

import pandas as pd


class MobileData:

    def __init__(self, process: Enum):
        """
        Initialize the MobileData instance.

        Args:
            process: An enum representing the process. The enum value should be the path to the CSV file.
        """
        dataframe = pd.read_csv(process.value)
        dataframe['time'] = pd.to_datetime(dataframe['time'], format='ISO8601')
        dataframe.set_index('time', inplace=True)
        dataframe.sort_index(inplace=True)
        self._df = dataframe[['Bending Moment', 'Axial Force', 'Torsion']]

    @property
    def df(self):
        """
        Get the dataframe containing the mobile data.

        Returns:
            A pandas DataFrame containing the columns 'Bending Moment', 'Axial Force', and 'Torsion'.
        """
        return self._df

    @df.setter
    def df(self, val):
        """
        Set the dataframe containing the mobile data.

        Args:
            val: A pandas DataFrame containing the columns 'Bending Moment', 'Axial Force', and 'Torsion'.
        """
        self._df = val
