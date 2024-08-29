import os
from fastdtw import fastdtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.classes.Utility import Utility


class QualityTest:

    def __init__(self, threshold=0.1):
        """
        Initialize the QualityTest instance.

        Args:
            threshold: The similarity score threshold for rejecting segments.
        """
        self._ground_truth = None
        self._threshold = threshold

    def _calc_similarity_score(self, segment):
        """
        Calculate the similarity score between a segment and the ground truth.

        Args:
            segment: The data segment to compare with the ground truth.

        Returns:
            The mean similarity score.
        """
        dtw_distances = []
        segment = Utility.scale_data(segment)
        gt = Utility.scale_data(self._ground_truth)
        for column in segment.columns:
            series1 = segment[column].values
            series2 = gt[column].values

            # Calculate the DTW distance and path
            distance, path = fastdtw(series1, series2, dist=2)

            similarity_score = distance / len(path)
            dtw_distances.append(similarity_score)

        return np.mean(dtw_distances)

    def _reject_false_segments(self, sim_score, data, segment_num, process):
        """
        Reject a segment if its similarity score is below the threshold.

        Args:
            sim_score: The similarity score of the segment.
            data: The dataset containing the segment.
            segment_num: The segment number.
            process: The process identifier.

        Returns:
            The updated dataset with the rejected segment.
        """
        if sim_score < self._threshold:
            pass_val = abs(segment_num)  # Make segment number positive
            data.loc[data['Segment Number'] == segment_num, 'Segment Number'] = pass_val
            print('File/Segment: ' + process + '/' + str(
                pass_val) + ' has passed due to similarity score of: ' + str(sim_score) + '.')
        return data

    def _process_file(self, full_path_src, full_path_target, initial_step):
        """
        Load and process a file, then save the results to the target path.

        Args:
            full_path_src: The source file path.
            full_path_target: The target file path.
            initial_step: If True, all segment numbers are negated to mark them as rejected.
        """
        # Load and process data
        data = self._load_data(full_path_src)
        if initial_step:
            data = self._mark_data_as_rejected(data)
        unique_segments = data['Segment Number'].unique()

        # Calculate similarity scores for each segment
        for segment_num in unique_segments:
            segment_data = data[data['Segment Number'] == segment_num]
            sim_score = self._calc_similarity_score(segment_data)
            data = self._reject_false_segments(sim_score, data, segment_num, os.path.basename(full_path_src))

        # save file to target
        data.to_csv(full_path_target, index=True)

    def run_fastdtw(self, source_path, file='all', target_path=None, initial_step=True):
        """
        Run the fast dynamic time warping (fastdtw) process on the data.

        Args:
            source_path: The directory containing the source files.
            file: Specific file(s) to process, or 'all' to process all files.
            target_path: The directory to save the processed files. If None, saves to source_path.
            initial_step: If True, all segment numbers are negated to mark them as rejected.
                        If False, only the similarity scores are calculated and segments that pass are made positive.
        """
        if target_path is None:
            target_path = source_path

        # Create the target directory if it does not exist
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        if file.lower() == 'all':
            # Process all CSV files in the source directory
            for filename in os.listdir(source_path):
                if filename.endswith('.csv'):
                    print('starting fastdtw for: ' + filename)
                    full_path_src = os.path.join(source_path, filename)
                    full_path_target = os.path.join(target_path, filename)
                    self._process_file(full_path_src, full_path_target, initial_step)
        else:
            # Process specific files separated by semicolons
            processes = file.lower().split(';')
            for process in processes:
                print('starting fastdtw for: ' + process)
                full_path_src = os.path.join(source_path, process + '.csv')
                full_path_target = os.path.join(target_path, process + '.csv')
                self._process_file(full_path_src, full_path_target, initial_step)

    def _mark_data_as_rejected(self, data):
        """
        Mark all segment numbers as rejected by negating them.

        Args:
            data: The dataset containing the segments.

        Returns:
            The updated dataset with all segments marked as rejected.
        """
        data['Segment Number'] = -data['Segment Number']
        return data

    def _load_data(self, source_path):
        """
        Load data from a CSV file and preprocess it.

        Args:
            source_path: The path to the source CSV file.

        Returns:
            The preprocessed data as a pandas DataFrame.
        """
        dataframe = pd.read_csv(source_path)
        dataframe['time'] = pd.to_datetime(dataframe['time'], format='ISO8601')
        dataframe.set_index('time', inplace=True)
        dataframe.sort_index(inplace=True)
        return dataframe

    def set_ground_truth(self, source_path, segment_num):
        """
        Set the ground truth data based on the specified segment number.

        Args:
            source_path: The path to the source file containing the ground truth.
            segment_num: The segment number to be used as the ground truth.
        """
        data = self._load_data(source_path)
        gt = data.loc[data['Segment Number'] == segment_num]
        self._ground_truth = gt

    def plot_rejected_segments(self, data):
        """
        Plot all rejected segments for visualization.

        Args:
            data: The dataset containing the rejected segments.
        """
        rejected_segments = data[data['Segment Number'] < 0]['Segment Number'].unique()

        for segment_num in rejected_segments:
            segment_data = data[data['Segment Number'] == segment_num]
            plt.figure(figsize=(12, 6))
            plt.plot(segment_data.index, segment_data['Bending Moment'], label='Bending Moment')
            plt.plot(segment_data.index, segment_data['Axial Force'], label='Axial Force')
            plt.plot(segment_data.index, segment_data['Torsion'], label='Torsion')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Rejected Segment: {abs(segment_num)}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    def set_threshold(self, threshold):
        """
        Set the threshold for similarity scoring.

        Args:
            threshold: The new threshold value.
        """
        self._threshold = threshold
