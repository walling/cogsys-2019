#!/bin/bash

# Exit script on first error.
set -e

# Word in root directory.
cd "$(dirname "$(dirname "$(dirname "$0")")")"

# Settings.
src_dir=../../data/3-experiment/graph
dst_dir=paper/img

# Copy generic images for all experiments.
for eid in 01 02 03 04; do
    echo "[experiment $eid] [generic]"
    ln -s -f    $src_dir/e${eid}_confusion_matrix_acoustic_r03.pdf \
                $dst_dir/e${eid}_cm_a.pdf
    ln -s -f    $src_dir/e${eid}_confusion_matrix_visual_r03.pdf \
                $dst_dir/e${eid}_cm_v.pdf
    ln -s -f    $src_dir/e${eid}_map_size.pdf \
                $dst_dir/e${eid}_size_graph.pdf
    ln -s -f    $src_dir/e${eid}_hit_map_acoustic_r03.pdf \
                $dst_dir/e${eid}_hit_map_a.pdf
    ln -s -f    $src_dir/e${eid}_hit_count_acoustic_r03.pdf \
                $dst_dir/e${eid}_hit_count_a.pdf
    ln -s -f    $src_dir/e${eid}_hit_map_visual_r03.pdf \
                $dst_dir/e${eid}_hit_map_v.pdf
    ln -s -f    $src_dir/e${eid}_hit_count_visual_r03.pdf \
                $dst_dir/e${eid}_hit_count_v.pdf
done

# Copy images for no hebbian experiments.
for eid in 01 02; do
    echo "[experiment $eid] [no hebbian]"
    ln -s -f    $src_dir/e${eid}_correct.pdf \
                $dst_dir/e${eid}_correct_graph.pdf
done

# Copy images for experiments using hebbian.
for eid in 03 04; do
    echo "[experiment $eid] [using hebbian]"
    ln -s -f    $src_dir/e${eid}_taxonomic_bar.pdf \
                $dst_dir/e${eid}_tax_bar.pdf
    ln -s -f    $src_dir/e${eid}_taxonomic_graph.pdf \
                $dst_dir/e${eid}_tax_graph.pdf
done
