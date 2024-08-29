import os
import multiprocessing
from src.classes.CPDetector import CPDetector
from src.classes.Utility import Utility
from src.classes.MobileData import MobileData
from src.classes.OverlappedChunking import OverlappedChunking
from src.classes.ChunkProcessor import ChunkProcessor
from src.classes.Chunk import Chunk
from ressources.enums.DrillingProcess import DrillingProcess
from ressources.enums.SmoothingProcess import SmoothingProcess
import gc
from ressources.exceptions.SegmentationError import SegmentationError


class SegmentationProcessor:
    """
    Initialize the SegmentationProcessor.

    Args:
        config: The configuration for segmentation.
        segment_column_name: The name of the segment column.
        cores: Number of CPU cores to use for processing.
    """
    def __init__(self, config, segment_column_name, cores=None):
        self._config = config
        self._segment_column_name = segment_column_name
        self._ov_chunking = OverlappedChunking()
        cores = self._define_cores(cores)
        self._cores = cores
        config_name, config_val = config
        self._output_path = config_val['target_path']
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

    def _define_cores(self, cores):
        """
        Define the number of CPU cores to use for processing.

        Args:
            cores: Desired number of CPU cores.

        Returns:
            Number of CPU cores to use.
        """
        my_num_cpus = multiprocessing.cpu_count()
        if cores is None:
            return max(my_num_cpus // 2, 1)
        if cores > my_num_cpus:
            raise SegmentationError('cpu number exceeded')
        return cores

    def process_data(self):
        """
        Process the data based on the configuration.
        """
        name, config = self._config
        cpd = CPDetector(config['model'], config['algorithm'], config['jump_points'],
                         config.get('min_segment_size'),
                         config.get('penalty_term'), config.get('model_parameters'), config.get('estimated_cps'),
                         config.get('window'))

        if config['process'].lower() == 'all':
            if name.lower() == 'drilling_config':
                self._process_all(DrillingProcess, cpd, config, name)
            if name.lower() == 'smoothing_config':
                self._process_all(SmoothingProcess, cpd, config, name)
        else:
            processes = config['process'].lower().split(';')
            if name.lower() == 'drilling_config':
                self._process_selected(DrillingProcess, processes, cpd, config, name)
            if name.lower() == 'smoothing_config':
                self._process_selected(SmoothingProcess, processes, cpd, config, name)

    def _process_all(self, process_enum, cpd, config, config_type):
        """
        Process all processes in the given enum.

        Args:
            process_enum: The enum of processes to process.
            cpd: The CPDetector instance.
            config: The configuration for segmentation.
            config_type: The type of configuration (drilling or smoothing).
        """
        for process in process_enum:
            self._process_single(process, cpd, config, config_type)

    def _process_selected(self, process_enum, processes, cpd, config, config_type):
        """
        Process the selected processes.

        Args:
            process_enum: The enum of processes.
            processes: List of process names to process.
            cpd: The CPDetector instance.
            config: The configuration for segmentation.
            config_type: The type of configuration (drilling or smoothing).
        """
        for process_name in processes:
            process = process_enum[process_name.upper()]
            self._process_single(process, cpd, config, config_type)

    def _process_single(self, process, cpd, config, config_type):
        """
        Process a single process.

        Args:
            process: The process to be segmented.
            cpd: The CPDetector instance.
            config: The configuration for segmentation.
            config_type: The type of configuration (drilling or smoothing).
        """
        print('Starting Segmentation of: ' + process.name)
        data = MobileData(process).df
        scaled_data = Utility.scale_data(data)
        chunks = self._ov_chunking.chunk_data(scaled_data, config['chunk_size'], config['overlap_region'], 0)
        print('Number of chunks created: ' + str(len(chunks)))
        c_processor = ChunkProcessor(chunks, cpd, self._cores)
        c_processor.process_chunks()
        results: list[Chunk] = c_processor.get_results()
        merged_cps = self._ov_chunking.merge_chunks(results)
        cpd_list = list(merged_cps)
        cpd_list.sort()
        if config['filter_close_cps'] is True:
            cpd_list = CPDetector.adaptive_mean_filter(scaled_data, cpd_list, config['min_cp_distance'])
        print('Changepoints found: ')
        print(cpd_list)
        data[self._segment_column_name] = 0
        current_segment = 1
        previous_cp = 0
        for cp in cpd_list:
            print(f"Assigning segment number {current_segment} from {previous_cp} to {cp}")
            data.iloc[previous_cp:cp, data.columns.get_loc(self._segment_column_name)] = current_segment
            current_segment += 1
            previous_cp = cp
        data.iloc[previous_cp:, data.columns.get_loc(self._segment_column_name)] = current_segment
        print(data[self._segment_column_name].value_counts())

        output_file = os.path.join(self._output_path, process.name + '.csv')

        data.to_csv(output_file, index=True)
        self._ov_chunking.reset_inkrement()

        del data, scaled_data, chunks, c_processor, results, merged_cps, cpd_list
        gc.collect()
