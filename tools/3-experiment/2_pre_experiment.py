#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.model.experiment import ExperimentRunner

# Setup experiment.
experiment = ExperimentRunner(
    experiment_id=2,
    map_size=(10, 10),
    map_expanding=True,
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
        2881569921,
        1161227264,
        2948391081,
        3500938763,
        2835782318,
        3557147914,
        1820185261,
        3578106459,
        4144414516,
        2821894926,
    ],
)

# Run the experiment.
experiment.run()
