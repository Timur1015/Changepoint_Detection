from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class Utility:

    def __init__(self):
        pass

    @staticmethod
    def scale_data(data: pd.DataFrame):
        """
        Scale the data using StandardScaler.

        Args:
            data: A pandas DataFrame to be scaled.

        Returns:
            Scaled data as a pandas DataFrame.
        """
        sc = StandardScaler().set_output(transform='pandas')
        return sc.fit_transform(data)

    @staticmethod
    def inverse_scaling(data: pd.DataFrame):
        scaler = StandardScaler()
        return scaler.inverse_transform(data)

    @staticmethod
    def get_penalty(time_series_len, pen_type: str, aprox_number_of_cps, modell_params):
        """
        Calculate the penalty value based on the specified penalty type.

        Args:
            time_series_len: The length of the time series.
            pen_type: The type of penalty to calculate.
            aprox_number_of_cps: The approximate number of change points.
            modell_params: Model parameters.

        Returns:
            The calculated penalty value.
        """
        p = modell_params  # location of the changepoints and value of meanshift and autocorellation (p should be 3)
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

    @staticmethod
    def plot_data(pd_obj, interval, process_name, process_type):
        """
        Plot the data using pandas' plot function.

        Args:
            pd_obj: A pandas DataFrame or Series to plot.
            interval: The time interval for the x-axis.
            process_name: The name of the process.
            process_type: The type of the process.
        """
        if isinstance(pd_obj, pd.DataFrame):
            column_list = pd_obj.columns
            plt_title = process_type + ' / ' + process_name + ' / ' + column_list[0]
            y_label = column_list[0]
            for i in range(1, len(column_list), 1):
                plt_title = plt_title + ' & ' + column_list[i]
                y_label = y_label + ',' + column_list[i]
        else:
            plt_title = process_type + ' / ' + process_name + ' / ' + pd_obj.name
            y_label = pd_obj.name
        if interval is None:
            pd_obj.plot(title=plt_title,
                        xlabel='Time',
                        ylabel=y_label,
                        figsize=(20, 10),
                        grid=True,
                        linewidth=1)

        else:
            pd_obj.plot(title=plt_title,
                        xlabel='Time',
                        ylabel=y_label,
                        figsize=(20, 10),
                        grid=True,
                        xlim=interval,
                        linewidth=1)

    @staticmethod
    def whiten_penalty_by_ac(pen, ac_value):
        """
        Inflate the penalty to consider autocorrelation of second order structure.

        Args:
            pen: The original penalty value.
            ac_value: The autocorrelation value.

        Returns:
            The inflated penalty value.
        """
        if ac_value >= 0:
            w_pen = pen * (1 + ac_value)
        else:
            w_pen = pen * (1 - ac_value)
        return w_pen

    @staticmethod
    def chunk_df_by_time(dataframe, time):
        """
        Chunk the DataFrame by time intervals.

        Args:
            dataframe: The pandas DataFrame to be chunked.
            time: The time interval for chunking.

        Returns:
            A list of chunked DataFrames.
        """
        interval = time
        return [group for _, group in dataframe.groupby(pd.Grouper(freq=interval))]

    @staticmethod
    def filter_cps_by_treshold(changepoints: list, treshold: int):
        """
        Filter change points by a specified threshold.

        Args:
            changepoints: A list of detected change points.
            treshold: The minimum distance threshold between change points.

        Returns:
            An array of filtered change points.
        """
        filtered_changepoints = [changepoints[0]]

        # check if distance between changepoints violates the treshold
        for i in range(1, len(changepoints)):
            if changepoints[i] - filtered_changepoints[-1] >= treshold:
                filtered_changepoints.append(changepoints[i])
            else:
                # if distance is
                filtered_changepoints[-1] = changepoints[i]

        return np.array(filtered_changepoints)

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
