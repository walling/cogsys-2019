#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Library imports.
from lib.data.experiment.analyze import ExperimentAnalyzer

# Run for all experiment logs.
ExperimentAnalyzer().run()
