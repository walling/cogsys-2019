
import pandas as pd
import numpy as np
from .. import file

def load_stimuli():
    stimuli = pd.read_csv(file.acoustic_path('stimuli.csv'), index_col=0)
    stimuli['vector'] = stimuli.vector.str.split(' ').apply(np.asarray, dtype=np.float)
    return stimuli
