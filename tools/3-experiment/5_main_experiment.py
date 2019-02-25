#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.model.experiment import ExperimentRunner

# Setup experiment.
experiment = ExperimentRunner(
    experiment_id=5,
    map_size=(10, 10),
    map_expanding=True,
    use_hebbian=True,
    epochs_count=100,       # TODO: Run for 300 epochs
    categories_count=33,    # TODO: Use all 100 categories
    variants_count=18,      # TODO: Use all 24 variants
    categories_new_count=9, # TODO: Use 10 new categories per round
    categories_retain_count=4,
    variant_types=['low', 'medium', 'high'],
    learning_rate_range=(0.7, 0.0002),
    neighborhood_update_range=(1.0, 0.25),
    neighborhood_insert_range=(0.1, 0.1),
    random_seeds=[
        763217585,
        2132962247,
        3075829152,
        2850768734,
        781280814,
        2060117787,
        3346885033,
        578902650,
        3205011335,
        3065817162,
    ],
)

# Run the experiment.
experiment.run()
