
from PIL import Image, ImageDraw
import numpy as np

def generate_prototype_dots(grid_size, dots_count):
    pixels = np.arange(grid_size[0] * grid_size[1])
    dots   = np.random.permutation(pixels)[0:dots_count]
    xs, ys = np.unravel_index(dots, shape=grid_size)
    return np.array(list(zip(xs, ys)))

def generate_variant_dots(prototype_dots, grid_size, sigma):
    prototype_dots = np.asarray(prototype_dots, dtype=np.float64)
    dots_count     = prototype_dots.shape[0]
    while True:
        variant_dots = np.round(prototype_dots
            + np.random.normal(scale=sigma, size=(dots_count, 2))).astype(int)

        if (np.min(variant_dots) < 0
                or np.any(np.max(variant_dots, axis=0) >= grid_size)):
            continue

        variant_dots = set(map(tuple, variant_dots))
        if len(variant_dots) != dots_count:
            continue

        return np.array(list(variant_dots))

def generate_all_variant_dots(prototype_dots, grid_size, variant_types):
    return np.array([generate_variant_dots(prototype_dots, grid_size, sigma)
                for count, sigma, label in variant_types for i in range(count)])

def dots_to_grid(dots, grid_size):
    grid = np.zeros(grid_size, dtype=int)
    np.put(grid, np.ravel_multi_index(dots.T, grid_size), 1)
    return grid

def dots_pattern_image(prototypes_dots, variants_dots, grid_size, offset=1, fillcolor=0):
    categories_count, variants_count = variants_dots.shape[0:2]

    field_width  = grid_size[0] + 3
    field_height = grid_size[1] + 3
    image_width  = (variants_count   + 2) * field_width  + 1
    image_height = (categories_count + 1) * field_height + 1

    img = Image.new(mode='1', size=(image_width, image_height))

    context = ImageDraw.Draw(img)
    context.rectangle([(0, 0), (image_width, image_height)], fill=1-fillcolor)

    for y in range(categories_count + 1):
        for x in range(variants_count + 2):
            context.line([
                ((x + 1) * field_width + 1, (y + 1) * field_height - 1),
                ((x + 2) * field_width - 3, (y + 1) * field_height - 1),
            ], fill=fillcolor)
            context.line([
                ((x + 1) * field_width - 1, (y + 1) * field_height + 1),
                ((x + 1) * field_width - 1, (y + 2) * field_height - 3),
            ], fill=fillcolor)
            if x >= variants_count + 1 or y >= categories_count: continue

            if x == 0:
                dots = prototypes_dots[y]
            else:
                dots = variants_dots[y][x - 1]

            xy = ((x + 1) * field_width + 1, (y + 1) * field_height + 1)
            for dot in dots:
                context.point((xy[0] + dot[0], xy[1] + dot[1]), fill=fillcolor)

    context.text((field_width + 2, 10), 'Proto', fill=fillcolor)
    for x in range(variants_count):
        label = '%02d' % (x + 1)
        context.text(((x + 2) * field_width + 11, 10), label, fill=fillcolor)
    for y in range(categories_count):
        label = '%03d' % (y + offset)
        context.text((7, (y + 1) * field_height + 10), label, fill=fillcolor)

    return img
