
import pandas as pd
import numpy as np
from pathlib import Path
from ..file import experiment_log_path, str_to_array

def load_log(experiment_id=None, filename=None):
    # Use `experiment_id` if given, otherwise filename as-is.
    if experiment_id is not None:
        filename = experiment_log_path('experiment_%d_log.csv' % experiment_id)

    # Load log file as data frame.
    log      = pd.read_csv(filename, index_col=0)

    # Fix array columns.
    log['acoustic_confusion_matrix'] = log.acoustic_confusion_matrix.apply(str_to_array)
    log['acoustic_hit_map']          = log.acoustic_hit_map.apply(str_to_array)
    log['acoustic_hit_count']        = log.acoustic_hit_count.apply(str_to_array)
    log['acoustic_error_field']      = log.acoustic_error_field.apply(str_to_array)
    log['visual_confusion_matrix']   = log.visual_confusion_matrix.apply(str_to_array)
    log['visual_hit_map']            = log.visual_hit_map.apply(str_to_array)
    log['visual_hit_count']          = log.visual_hit_count.apply(str_to_array)
    log['visual_error_field']        = log.visual_error_field.apply(str_to_array)

    # Return log data frame.
    return log

def load_all_logs():
    logs = [load_log(filename=log_file)
        for log_file in Path(experiment_log_path()).glob('experiment_*_log.csv')]
    return pd.concat(logs, ignore_index=True)
