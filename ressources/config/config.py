# File includes the metadata for processing the files

# Configuration for segmentation process

# Configuration for drilling process
drilling_config = {
    'process': 'PROCESS_23;PROCESS_28',  # Processes to be handled. To handle all existing processes set 'all'
    'target_path': '../data/segmented/drilling_data',  # Path to save segmented data
    'estimated_cps': 360,  # Estimated change points (180 drills * 2)
    'model': 'l2',  # Model type
    'model_parameters': 2,  # Model parameters
    'penalty_term': 'BIC',  # Penalty term for model selection
    'algorithm': 'PELT',  # Algorithm used for segmentation
    'min_segment_size': 500,  # Minimum segment size
    'jump_points': 50,  # Jump points in data
    'chunk_size': 40000,  # Chunk size for processing
    'overlap_region': 300,  # Overlap region size
    'min_cp_distance': 1400,  # Minimum change point distance
    'filter_close_cps': True  # Whether to filter close change points
}

smoothing_config = {
    'process': 'PROCESS_26',  # Processes to be handled. To handle all existing processes set 'all'
    'target_path': '../data/segmented/smoothing_data',  # Path to save segmented data
    'estimated_cps': 360,  # Estimated change points (180 smoothings * 2)
    'model': 'rbf',  # Model type
    'model_parameter': 2,  # Model parameters
    'penalty_term': 'Hannan Quinn',  # Penalty term for model selection
    'algorithm': 'KernelCPD',  # Algorithm used for segmentation
    'min_segment_size': 1000,  # Minimum segment size
    'jump_points': 500,  # Jump points in data
    'chunk_size': 10000,  # Chunk size for processing
    'overlap_region': 1000,  # Overlap region size
    'filter_close_cps': False  # Whether to filter close change points
}

# List of segmentation configurations
seg_config = [
    ('drilling_config', drilling_config),
    ('smoothing_config', smoothing_config)
]

# Configuration for testing

# Enable or disable testing mode
testing_enabled = False

# Configuration for drilling test
drill_test_config = {
    'gt_source_path': '../data/segmented/drilling_data/PROCESS_23.csv',  # Ground truth source path
    'gt_seg_nums': [3, 4],  # Ground truth segment numbers
    'gt_thresholds': [0.25, 0.4],  # Ground truth thresholds
    'exec_type': 'manually',
    # Execution type (manually/auto). If auto is set than testing is based on previously executed segmentation
    'process': 'PROCESS_23',  # Process to be tested
    'target_path': '../data/tested/drilling_data'  # Path to save test results
}

# Configuration for smoothing test
smooth_test_config = {
    'gt_source_path': '../data/segmented/smoothing_data/PROCESS_26.csv',  # Ground truth source path
    'gt_seg_nums': [8, 9],  # Ground truth segment numbers
    'gt_thresholds': [0.4, 0.4],  # Ground truth thresholds
    'exec_type': 'manually',
    # Execution type (manually/auto). If auto is set than testing is based on previously executed segmentation
    'process': 'PROCESS_26',  # Process to be tested
    'target_path': '../data/tested/smoothing_data'  # Path to save test results
}

# List of test configurations
test_config = [
    ('drill_test_config', drill_test_config),
    ('smooth_test_config', smooth_test_config)
]
