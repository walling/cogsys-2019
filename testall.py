#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0].resolve()))

# System imports.
import doctest
import lib
import lib.data
import lib.data.file
import lib.data.visual
import lib.data.visual.generate
import lib.model
import lib.model.associative_memory
import lib.model.gsom
import lib.model.hebbian
import lib.model.math_functions
import lib.settings

# List modules to test.
modules = [
    lib,
    lib.data,
    lib.data.file,
    lib.data.visual,
    lib.data.visual.generate,
    lib.model,
    lib.model.associative_memory,
    lib.model.gsom,
    lib.model.hebbian,
    lib.model.math_functions,
    lib.settings,
]

# Helper function to test all modules.
def testall():
    # Test each module and count number of failures.
    failed_modules = 0
    for module in modules:
        result = doctest.testmod(module)
        if result.failed > 0:
            failed_modules += 1
            print()

    # Output summary and exit with appropriate status code.
    if failed_modules == 0:
        print('All %d modules passed.' % len(modules))
        exit(0)
    else:
        print('%d out of %d modules failed.' % (failed_modules, len(modules)))
        exit(1)

# When invoked as a script, run all tests.
if __name__ == '__main__':
    testall()
