from ressources.exceptions.SegmentationError import SegmentationError
import ruptures as rpt
import numpy as np
import warnings


class CPDetector:
    def __init__(self, model, algorithm_name, jump_points, min_seg_size=None, penalty=None, model_params=None,
                 n_cps=None, window=None):
        """
        Initialize the CPDetector.

        Args:
            model: The model used for change point detection.
            algorithm_name: The name of the algorithm to use.
            jump_points: The number of jump points.
            min_seg_size: Minimum segment size.
            penalty: The penalty term for model selection.
            model_params: Parameters for the model.
            n_cps: Number of change points to detect.
            window: Window size for the detection.
         """
        self._model = model
        self._model_params = model_params
        self._penalty = penalty
        self._n_cps = n_cps
        self._window = window
        self._algorithm_map = {
            'pelt': 'Pelt',
            'kernelcpd': 'KernelCPD',
            'dynp': 'Dynp',
            'binseg': 'Binseg',
            'bottomup': 'BottomUp',
            'window': 'Window'
        }
        self._sanity_check()
        self._algorithm = self.map_algorithm_by_name(algorithm_name, min_seg_size, jump_points)

    def map_algorithm_by_name(self, algorithm_name, min_seg_size, jump_points):
        """
        Map the algorithm name to the corresponding ruptures algorithm.

        Args:
            algorithm_name: The name of the algorithm to map.
            min_seg_size: Minimum segment size.
            jump_points: Number of jump points.

        Returns:
            The mapped algorithm instance.
        """
        algorithm_name = algorithm_name.lower()
        algorithm_name = self._algorithm_map.get(algorithm_name)
        try:
            algo = getattr(rpt, algorithm_name)
        except AttributeError:
            raise SegmentationError('Algorithm class not defined')
        if algorithm_name != 'KernelCPD':
            if self._window is None:
                return algo(model=self._model, min_size=min_seg_size, jump=jump_points)
            else:
                return algo(width=self._window, model=self._model, jump=jump_points)
        else:
            return algo(kernel=self._model, min_size=min_seg_size, jump=jump_points)

    def run(self, data):
        """
        Run the change point detection algorithm on the data.

        Args:
            data: The data to run the algorithm on.

        Returns:
            List of detected change points.
        """
        self._algorithm.fit(data.values)
        if self._penalty is not None:
            pen = self._get_penalty_value(len(data), self._penalty, self._n_cps, self._model_params)
            cps = self._algorithm.predict(pen=pen)
        else:
            cps = self._algorithm.predict(n_bkps=self._n_cps)
        return cps

    def _get_penalty_value(self, time_series_len, pen_type: str, aprox_number_of_cps, modell_params):
        """
        Get the penalty value for the specified penalty type.

        Args:
            time_series_len: The length of the time series.
            pen_type: The type of penalty.
            aprox_number_of_cps: The approximate number of change points.
            modell_params: Model parameters.

        Returns:
            The penalty value.
        """
        p = modell_params  # location of the changepoints and value of meanshift (should be 2)
        _t = aprox_number_of_cps
        pen_type = pen_type.lower()
        penalty_method_dict = {'sic': p * _t * np.log(time_series_len),
                               'bic': p * _t * np.log(time_series_len),
                               'aic': p * _t * 2,
                               'hannan quinn': 2 * p * _t * np.log(np.log(time_series_len))}
        pen = penalty_method_dict[pen_type]
        if pen is None:
            raise ValueError(f"Penalty type '{pen_type}' not supported")
        else:
            return pen

    """
        the method 'adaptive_mean_filter' describes a domain-specific rule that is used on the
        results by a changepoint-method. The results of the changepoint-method might be statistically correct, but it might
        not match our domain-specific goal of segmentation.Aside of that working with changepoint detection methods with
        d>1 dimensions does not consider dependencies between dimensions. However a robust strategy to work around this
        issue is to exceed the segmenation process within 2 steps. Step 1: statistically correct Segmentation Step 2: domain
        specific filtering. Step 2 can be achieved by using the 'adaptive_mean_filter'-method, which looks at changepoints
        where it's distance can be considered as too close to each other. The determination of the specific distance, which
        is described as the threshold requires knowledge about the data itself. Then the method calculates the change in
        mean for the segment for each of the changepoints. If the change in mean rises, than we can assume that the current
        location in the data must be at the start of a segment. This mean that the earlier point is correct and the later
        point is wrong.This assumption is true because most of the wrong detected changepoints are located at the start or 
        the end of a segment. The reason for that could be a lack of synchronicity due to latency or other mechanically
        reasons
        """

    @staticmethod
    def adaptive_mean_filter(data, changepoints, treshold: int):
        dummy_cp = 0
        filtered_changepoints = [changepoints[0]]

        for i in range(1, len(changepoints)):
            p1 = filtered_changepoints[-1]
            p2 = changepoints[i]
            if p2 - p1 < treshold:
                if len(filtered_changepoints) < 2:
                    z = dummy_cp
                else:
                    z = filtered_changepoints[-2]
                if np.mean(data[z:p2]) > np.mean(data[z:p1]):
                    filtered_changepoints[-1] = p1
                else:
                    filtered_changepoints[-1] = p2
            else:
                filtered_changepoints.append(p2)
        return np.array(filtered_changepoints)

    def _sanity_check(self):
        """
        Perform a sanity check on the parameters.

        Raises:
            SegmentationError: If necessary parameters are missing or inconsistent.
        """
        if self._penalty is None and self._n_cps is None:
            raise SegmentationError("Either penalties or n_cps must be provided.")
        if self._penalty is not None and self._model_params is None:
            self._model_params = 2
        if self._penalty is None and self._model_params is None:
            warnings.warn('Penalty is not defined, so model parameters are ignored.')

    def set_n_cps(self, n_cps):
        """
        Set the number of change points to detect.

        Args:
            n_cps: The number of change points.
        """
        self._n_cps = n_cps

    def get_n_cps(self):
        """
        Get the number of change points to detect.

        Returns:
            The number of change points.
        """
        return self._n_cps
