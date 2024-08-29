import os

from ressources.config.config import seg_config, test_config, testing_enabled
from src.classes.SegmentationProcessor import SegmentationProcessor
from tests.classes.QualityTest import QualityTest
from ressources.exceptions.SegmentationError import SegmentationError

if __name__ == '__main__':
    quality_test = QualityTest()

    # Iterate through the segmentation configurations
    for i, s_conf in enumerate(seg_config):
        # Initialize the segmentation processor with the current configuration
        seg_proc = SegmentationProcessor(s_conf, 'Segment Number', None)
        seg_proc.process_data()  # Process the data based on the segmentation configuration

        if testing_enabled:
            # Get the test configuration for the current segmentation configuration
            test_config_type = test_config[i][1]
            seg_config_type = s_conf[1]
            processes = seg_config_type['process']
            gt_seg_nums = test_config_type['gt_seg_nums']
            gt_thresholds = test_config_type['gt_thresholds']
            source_path = seg_config_type['target_path']
            target_path = test_config_type['target_path']

            # Ensure ground truth information is provided
            if gt_seg_nums is None or gt_thresholds is None or test_config_type['gt_source_path'] is None:
                raise SegmentationError('For testing a ground truth has to be provided')

            # Ensure the number of thresholds matches the number of ground truth segments
            if len(gt_thresholds) != len(gt_seg_nums):
                raise SegmentationError('You cannot have more thresholds than ground truths')

            # If the execution type is manually set, use the processes defined in the test configuration
            if test_config_type['exec_type'] == 'manually':
                # then the processes are set in test_config
                processes = test_config_type['process']

            # Iterate through the ground truth segment numbers and run the quality test
            quality_test.set_ground_truth(test_config_type['gt_source_path'], gt_seg_nums[0])
            quality_test.set_threshold(gt_thresholds[0])
            quality_test.run_fastdtw(source_path, processes, target_path, initial_step=True)
            source_path = target_path
            for j in range(1, len(gt_seg_nums)):
                quality_test.set_ground_truth(test_config_type['gt_source_path'], gt_seg_nums[j])
                quality_test.set_threshold(gt_thresholds[j])
                quality_test.run_fastdtw(source_path, processes, target_path, initial_step=False)
