'''
Settings to run the experiments.

Invariants
----------

>>> variants_count == sum(map(lambda t: t[0], visual_variant_types))
True
>>> variants_count == sum(map(lambda t: t[0], acoustic_variant_types))
True
>>> variants_count == len(visual_variant_types_by_index)
True
>>> variants_count == len(acoustic_variant_types_by_index)
True
'''

categories_count = 100 # this many prototypes will be generated
variants_count   = 24  # distortions made of each prototype

visual_stimulus_size = (30, 30) # width; height
visual_stimulus_dots = 9 # number of dots for each stimulus
visual_variant_types = [ # count; gaussian width (sigma) of distortion; label
    (8, 1.0, 'low'),
    (8, 2.0, 'medium'),
    (8, 3.0, 'high'),
]

acoustic_stimulus_size = (7, 4) # number of Mel bins; number of samples
acoustic_variant_types = [ # count; condition as Pandas query; label
    (19, 'speaker_id == 1', 'low'),
    (5,  'speaker_id != 1', 'high'),
]

visual_variant_types_by_index = [(count, sigma, label)
    for count, sigma, label in visual_variant_types for i in range(count)]

acoustic_variant_types_by_index = [(count, cond, label)
    for count, cond, label in acoustic_variant_types for i in range(count)]

def visual_variant_type_label(index):
    return visual_variant_types_by_index[index][2]

def acoustic_variant_type_label(index):
    return acoustic_variant_types_by_index[index][2]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
