
# System imports.
import os
import sys
from subprocess import call
from os.path import basename
import numpy as np

def root_path(*filename):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(script_dir)), *filename)

def data_path(*filename):
    return root_path('data', *filename)

def visual_path(*filename):
    return data_path('1-visual', *filename)

def visual_generated_pattern_path(*filename):
    return visual_path('generated', 'pattern', *filename)

def visual_generated_graph_path(*filename):
    return visual_path('generated', 'graph', *filename)

def acoustic_path(*filename):
    return data_path('2-acoustic', *filename)

def acoustic_original_wordlist_path(*filename):
    return acoustic_path('original', '1-wordlist', *filename)

def acoustic_original_audio_path(*filename):
    return acoustic_path('original', '2-audio', *filename)

def acoustic_processed_wordlist_path(*filename):
    return acoustic_path('processed', 'wordlist', *filename)

def acoustic_processed_audio_path(*filename):
    return acoustic_path('processed', 'audio', *filename)

def acoustic_processed_graph_path(*filename):
    return acoustic_path('processed', 'graph', *filename)

def experiment_path(*filename):
    return data_path('3-experiment', *filename)

def experiment_log_path(*filename):
    return experiment_path('log', *filename)

def experiment_graph_path(*filename):
    return experiment_path('graph', *filename)

def array_to_str(array):
    return ' '.join(map(str, np.ravel(array)))

def str_to_array(str):
    return np.asarray(str.split(' '), dtype=np.float64)

def optimize_png_size(filename):
    try:
        call(['optipng', '-quiet', '-o2', filename])
    except FileNotFoundError:
        print('Please install optipng to optimize PNG: %s'
            % basename(filename), file=sys.stderr)
