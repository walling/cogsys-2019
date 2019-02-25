#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.model.experiment import ExperimentRunner

# Setup experiment.
experiment = ExperimentRunner(
    experiment_id=1,
    map_size=(30, 30),
    map_expanding=False,
    use_hebbian=False,
    epochs_count=20,
    categories_count=30,
    variants_count=12,
    categories_new_count=10,
    categories_retain_count=4,
    variant_types=['low', 'medium'],
    learning_rate_range=(0.7, 0.0002),
    neighborhood_update_range=(1.0, 0.25),
    neighborhood_insert_range=(0.1, 0.1),
    random_seeds=[
        4240156900,
        1819592776,
        775546034,
        1810256561,
        1882592916,
        1556706631,
        427731614,
        1396917288,
        596050715,
        816706926,
    ],
)

# Run the experiment.
experiment.run()
