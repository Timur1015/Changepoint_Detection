import pandas as pd
from ruptures.metrics import hausdorff, precision_recall, randindex
from sklearn.metrics import normalized_mutual_info_score
from ressources.exceptions.SegmentationError import SegmentationError
from src.classes.Utility import Utility
import time
import numpy as np


class MetricSummary:

    def __init__(self):
        """
        Initialize the MetricSummary instance.
        """
        self.columns = ['Algorithm', 'Known CPs', 'Penalty', 'n_cps',
                        'Annotation Error', 'Hausdorff',
                        'Precision',
                        'Recall', 'F1', 'Randindex', 'NMI', 'Runtime']
        self._df = pd.DataFrame(columns=self.columns)
        self._margin = 0

    def add_row(self, row: list):
        """
        Add a row of metrics to the DataFrame.

        Args:
            row: A list containing the metric values.
        """
        n_rows = len(self._df)
        self._df.loc[n_rows] = row

    def calc_metrics(self, predicted_cp, ground_truth, data, margin_percent):
        """
        Calculate various metrics based on the predicted and ground truth change points.

        Args:
            predicted_cp: List of predicted change points.
            ground_truth: List of ground truth change points.
            data: The dataset being analyzed.
            margin_percent: The margin percentage for precision/recall calculation.

        Returns:
            A tuple of calculated metrics.
        """
        an_error = abs(len(predicted_cp) - len(ground_truth))  # annotation error
        hd = hausdorff(predicted_cp, ground_truth)  # hausdorff-metric
        self._margin = self._calc_margin_by_percent(data, margin_percent)
        p, r = precision_recall(ground_truth, predicted_cp, margin=self._margin)  # precision/recall with margin
        f1 = self._calc_f1_score(p, r)  # f1-score
        ri = randindex(predicted_cp, ground_truth)  # randindex
        nmi = self._calc_nmi(predicted_cp, ground_truth, len(data))  # normalized mutual info score
        return an_error, hd, p, r, f1, ri, nmi

    def _calc_f1_score(self, p, r):
        """
        Calculate the F1 score.

        Args:
            p: Precision.
            r: Recall.

        Returns:
            The F1 score.
        """
        if p == 0 and r == 0:
            return 0
        else:
            return 2 * ((p * r) / (p + r))

    def _generate_labels(self, cps, n_samples):
        """
        Generate labels for each segment based on the change points.

        Args:
            cps: List of change points.
            n_samples: Number of samples in the dataset.

        Returns:
            A numpy array of labels.
        """
        labels = np.zeros(n_samples, dtype=int)
        current_label = 0
        prev_break = 0
        for break_point in cps:
            labels[prev_break:break_point] = current_label
            current_label += 1
            prev_break = break_point
        labels[prev_break:n_samples] = current_label
        return labels

    def _calc_nmi(self, predicted_cp, ground_truth, n_samples):
        """
        Calculate the Normalized Mutual Information (NMI) score.

        Args:
            predicted_cp: List of predicted change points.
            ground_truth: List of ground truth change points.
            n_samples: Number of samples in the dataset.

        Returns:
            The NMI score.
        """
        labels_true = self._generate_labels(ground_truth, n_samples)
        labels_pred = self._generate_labels(predicted_cp, n_samples)
        return normalized_mutual_info_score(labels_true, labels_pred)

    def compare_cpd_algorithms(self, data, ground_truth: list, algorithm: list, penalties: list = None,
                               n_cps: int = None,
                               metadata: dict = None):
        """
        Compare different change point detection algorithms.

        Args:
            data: The dataset being analyzed.
            ground_truth: List of ground truth change points.
            algorithm: List of algorithms to compare.
            penalties: List of penalties to apply (optional).
            n_cps: Number of change points to detect (optional).
            metadata: Additional metadata for the algorithms (optional).

        Raises:
            SegmentationError: If neither penalties nor n_cps are provided.
        """

        est_cp = 1
        mod_params = 2
        static_data = []
        if penalties is None and n_cps is None:
            raise SegmentationError("Either penalties or n_cps must be provided.")
        if metadata is not None:
            if 'estimated cps' in metadata:
                est_cp = metadata.pop('estimated cps')
            if 'model params' in metadata:
                mod_params = metadata.pop('model params')
            static_data = self._generate_dyn_columns(metadata)
        for algo in algorithm:
            algo_name = algo.__class__.__name__
            algo_spec_data = static_data.copy()
            algo_spec_data.append(algo_name)
            algo.fit(data.values)
            if penalties is not None:
                for pen in penalties:
                    spec_data = self._gen_penalized_stat(data, algo, ground_truth, pen, est_cp, mod_params)
                    stat_data = algo_spec_data.copy()
                    stat_data.extend(spec_data)
                    self.add_row(stat_data)
            else:
                spec_data = self._gen_cp_stat(data, algo, ground_truth, n_cps)
                algo_spec_data.extend(spec_data)
                self.add_row(algo_spec_data)

    def _generate_dyn_columns(self, metadata):
        """
        Generate dynamic columns for the DataFrame based on the provided metadata.

        Args:
            metadata: A dictionary containing metadata.

        Returns:
            A list of metadata values.
        """
        content = []
        for i, key in enumerate(metadata.keys()):
            if key not in self.columns:
                self.columns.insert(i, key)
                content.append(metadata[key])
        self._df = pd.DataFrame(columns=self.columns)
        return content

    def _gen_penalized_stat(self, data, algo, ground_truth, pen_name, est_cp, mod_params):
        """
        Generate statistics for a penalized change point detection algorithm.

        Args:
            data: The dataset being analyzed.
            algo: The algorithm being used.
            ground_truth: List of ground truth change points.
            pen_name: The name of the penalty.
            est_cp: Estimated number of change points.
            mod_params: Model parameters.

        Returns:
            A list of calculated metrics.
        """
        spec_data = [False, pen_name, 'unknown']
        start_time = time.time()
        cps = algo.predict(pen=Utility.get_penalty(len(data), pen_name, est_cp, mod_params))
        duration = time.time() - start_time
        metrics = self.calc_metrics(cps, ground_truth, data, 0.05)
        spec_data.extend(metrics)
        spec_data.append(duration)
        return spec_data

    def _gen_cp_stat(self, data, algo, ground_truth, n_cps):
        """
        Generate statistics for a change point detection algorithm with a fixed number of change points.

        Args:
            data: The dataset being analyzed.
            algo: The algorithm being used.
            ground_truth: List of ground truth change points.
            n_cps: Number of change points to detect.

        Returns:
        A list of calculated metrics.
        """
        spec_data = [True, n_cps]
        start_time = time.time()
        cps = algo.predict(n_bkps=n_cps)
        duration = time.time() - start_time
        metrics = self.calc_metrics(cps, ground_truth, data, 0.05)
        spec_data.extend(metrics)
        spec_data.append(duration)
        return spec_data

    def _calc_margin_by_percent(self, data, margin_percent):
        """
        Calculate the margin by a percentage of the data duration.

        Args:
            data: The data on which to calculate the margin.
            margin_percent: The margin percentage.

        Returns:
            The calculated margin.
        """
        start = data.index[0]
        end = data.index[len(data) - 1]
        duration = (end - start).total_seconds()  # duration of the process in seconds
        dp_per_seconds = len(data) / duration
        return dp_per_seconds * margin_percent

    @property
    def df(self):
        """
        Get the DataFrame containing the metric summaries.

        Returns:
            The DataFrame with metric summaries.
        """
        return self._df

    @property
    def size(self):
        """
        Get the number of rows in the metric summary DataFrame.

        Returns:
            The number of rows in the DataFrame.
        """
        return len(self._df)
