#!/usr/bin/env python3

# Setup import path to be able to import lib.* from repository root.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].resolve()))

# Import matplotlib before anything else to ensure PDF mode (without GUI).
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

# System imports.
from os.path import basename
import numpy as np
import pandas as pd

# Library imports.
from lib.data.visual import generate as gen
from lib.data import file
from lib import settings

# Fix seed at a chosen random value, to make this script deterministic, i.e. it
# can be run many times producing the same end result.
np.random.seed(2105805220)

# Load settings.
grid_size     = settings.visual_stimulus_size
dots_count    = settings.visual_stimulus_dots
variant_types = settings.visual_variant_types

# Generate prototype dots with output shape (100, 9, 2), i.e. 100 categories
# with 9 dots with (x,y) coordinates.
print('[tools:visual:generate_patterns] [generating_prototypes]')
prototypes_dots = np.array([
    gen.generate_prototype_dots(grid_size, dots_count)
        for i in range(settings.categories_count)])

# From the prototypes, generate all variants according to type specifications in
# settings file. The output shape is (100, 24, 9, 2): categories, variants,
# dots, (x,y) coordinates.
print('[tools:visual:generate_patterns] [generating_variants]')
variants_dots = np.array([
    gen.generate_all_variant_dots(prototype_dots, grid_size, variant_types)
        for prototype_dots in prototypes_dots])

# Prepare CSV data file.
print('[tools:visual:generate_patterns] [saving_csv]')
csv_rows = []
for category_index in range(settings.categories_count):
    category_id = category_index + 1
    dots        = prototypes_dots[category_index]
    csv_rows.append((
        'V_%03d_PT' % category_id,                            # id
        category_id,                                          # category_id
        'n/a',                                                # variant_id
        'prototype',                                          # variant_type
        file.array_to_str(gen.dots_to_grid(dots, grid_size)), # vector
    ))

    for variant_index in range(settings.variants_count):
        variant_id = variant_index + 1
        dots       = variants_dots[category_index][variant_index]
        csv_rows.append((
            'V_%03d_%02d' % (category_id, variant_id),            # id
            category_id,                                          # category_id
            variant_id,                                           # variant_id
            settings.visual_variant_type_label(variant_index),    # variant_type
            file.array_to_str(gen.dots_to_grid(dots, grid_size)), # vector
        ))

# Save and display CSV data.
csv_data = pd.DataFrame.from_records(data=csv_rows, index='id', columns=[
    'id',
    'category_id',
    'variant_id',
    'variant_type',
    'vector',
])
csv_data.query('variant_type == "prototype"').to_csv(file.visual_path('prototypes.csv'))
csv_data.query('variant_type != "prototype"').to_csv(file.visual_path('stimuli.csv'))
print()
print(csv_data)
print()

# Add all variants together to see the distribution visually.
variant_vectors = np.stack(
    csv_data
    .query('variant_type != "prototype"')
    .vector
    .apply(file.str_to_array)
)
dist_avg = np.mean(variant_vectors, axis=0).reshape(grid_size)
dist_std = np.std(variant_vectors, axis=0).reshape(grid_size)
def plot(filename, im, cmap):
    fig_filename = file.visual_generated_graph_path(filename)
    plt.ioff()
    plt.figure(figsize=(7, 7))
    plt.imshow(im, interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
    plt.close('all')
plot('distribution_avg.pdf', dist_avg, cmap=plt.cm.Greys)
plot('distribution_std.pdf', dist_std, cmap=plt.cm.Reds)

# Keep list of PNG files to optimize later.
png_files_to_optimize = []

# Generate PNG image with all prototype and variant dots for debugging.
img = gen.dots_pattern_image(prototypes_dots, variants_dots, grid_size)
filename = file.visual_generated_pattern_path('all_dots.png')
print('[tools:visual:generate_patterns] [saving_image] - %s' % basename(filename))
img.save(filename, format='PNG', optimize=True)
png_files_to_optimize.append(filename)

# Generate PNG images with prototype and variant dots, 10 categories at a time.
step_size = 10
for offset in range(0, settings.categories_count, step_size):
    img = gen.dots_pattern_image(prototypes_dots[offset:offset+step_size],
        variants_dots[offset:offset+step_size], grid_size, offset + 1)
    filename = file.visual_generated_pattern_path(
        '%03d_to_%03d_dots.png' % (offset + 1, offset + step_size))
    print('[tools:visual:generate_patterns] [saving_image] - %s' % basename(filename))
    img.save(filename, format='PNG', optimize=True)
    png_files_to_optimize.append(filename)

# Optimize all PNG files.
for filename in png_files_to_optimize:
    print('[tools:visual:generate_patterns] [optimizing] - %s' % basename(filename))
    file.optimize_png_size(filename)
