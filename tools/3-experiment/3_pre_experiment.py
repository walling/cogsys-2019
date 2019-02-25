#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.model.experiment import ExperimentRunner

# Setup experiment.
experiment = ExperimentRunner(
    experiment_id=3,
    map_size=(30, 30),
    map_expanding=False,
    use_hebbian=True,
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
        3941052043,
        2400130197,
        2562027226,
        1625778516,
        2602279308,
        4039331380,
        1417995820,
        3816341755,
        735272862,
        3645082649,
    ],
)

# Run the experiment.
experiment.run()
