#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.model.experiment import ExperimentRunner

# Setup experiment.
experiment = ExperimentRunner(
    experiment_id=4,
    map_size=(10, 10),
    map_expanding=True,
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
        3289915747,
        1233770346,
        3761763398,
        606616613,
        3090637167,
        1977998037,
        156678426,
        371987591,
        2334284321,
        1924123571,
    ],
)

# Run the experiment.
experiment.run()
