
import numpy as np
import pandas as pd
from .associative_memory import AssociativeMemory
from ..data import visual
from ..data import acoustic
from ..data.file import experiment_log_path, array_to_str

def print_matrix(matrix):
    '''Display integer matrices in terminal nicely.'''
    print('\n'.join(
        ['  ' + ' '.join([col and '%3d' % col or '  .' for col in row])
            for row in matrix]))

class ExperimentRunner:

    def __init__(self, experiment_id, map_size=(10, 10), map_expanding=True,
            use_hebbian=True, epochs_count=300, categories_count=100,
            variants_count=24, categories_new_count=10,
            categories_retain_count=4, variant_types=['low', 'medium', 'high'],
            learning_rate_range=(0.7,0.7), neighborhood_update_range=(1.0,1.0),
            neighborhood_insert_range=(0.1,0.1), random_seeds=[1]):
        # Load visual and acoustic data sets and filter according to setup.
        filtering = 'category_id <= %d' % categories_count
        filtering += ' & variant_id <= %d' % variants_count
        if variant_types is not None:
            filtering += ' & variant_type in %r' % variant_types
        stimuli_visual   = visual.load_stimuli().query(filtering)
        stimuli_acoustic = acoustic.load_stimuli().query(filtering)

        # Check the id columns of the visual and acoustic data sets match.
        ids_visual   = set([i[2:] for i in stimuli_visual.index])
        ids_acoustic = set([i[2:] for i in stimuli_acoustic.index])
        if ids_visual != ids_acoustic:
            raise Exception('Id mismatch of visual and acoustic stimuli'
                + (' - #visual: %d - #acoustic: %d' % (
                    len(ids_visual),
                    len(ids_acoustic))))

        # Rename acoustic data to prepare for linking with visual dataset.
        self._stimuli = stimuli = stimuli_acoustic
        stimuli['acoustic_vector'] = stimuli_acoustic.vector
        stimuli['acoustic_id']     = stimuli_acoustic.index.copy()
        stimuli['acoustic_type']   = stimuli_acoustic.variant_type

        # Link visual and acoustic datasets.
        stimuli.index            = stimuli_visual.index
        stimuli['visual_vector'] = stimuli_visual.vector
        stimuli['visual_id']     = stimuli_visual.index
        stimuli['visual_type']   = stimuli_visual.variant_type
        stimuli.reset_index(drop=True, inplace=True)

        # Prepare evaluation sets.
        self._acoustic_categories = np.stack(
            stimuli
            .groupby('category_id')
            .acoustic_vector
            .apply(np.stack)
            .to_frame()
            .acoustic_vector
            .to_numpy())
        self._visual_categories = np.stack(
            stimuli
            .groupby('category_id')
            .visual_vector
            .apply(np.stack)
            .to_frame()
            .visual_vector
            .to_numpy())

        # Calculate number of rounds for each experiment.
        self._rounds_count = int(np.ceil(categories_count/categories_new_count))

        # Store experiment parameters.
        self._experiment_id             = experiment_id
        self._map_size                  = map_size
        self._map_expanding             = map_expanding
        self._use_hebbian               = use_hebbian
        self._epochs_count              = epochs_count
        self._variants_count            = variants_count
        self._categories_count          = categories_count
        self._categories_new_count      = categories_new_count
        self._categories_retain_count   = categories_retain_count
        self._learning_rate_range       = learning_rate_range
        self._neighborhood_update_range = neighborhood_update_range
        self._neighborhood_insert_range = neighborhood_insert_range

        # Save random seeds for running the experiments.
        self._random_seeds = random_seeds

        # Create log object to store evaluation logs.
        self._log = pd.DataFrame(columns=[
            'experiment_id',
            'run_id',
            'round_id',
            'map_initial_width',
            'map_initial_height',
            'map_initial_size',
            'map_expanding',
            'categories_count',
            'categories_new_count',
            'categories_retain_count',
            'variants_count',
            'epochs_count',
            'taxonomic_factor',
            'acoustic_width',
            'acoustic_height',
            'acoustic_size',
            'acoustic_correct',
            'acoustic_generalization',
            'visual_width',
            'visual_height',
            'visual_size',
            'visual_correct',
            'visual_generalization',
            'acoustic_confusion_matrix',
            'acoustic_hit_map',
            'acoustic_hit_count',
            'acoustic_error_field',
            'visual_confusion_matrix',
            'visual_hit_map',
            'visual_hit_count',
            'visual_error_field',
        ])

    def run(self):
        # Setup a fixed random seed for each experiment and run it.
        for i, random_seed in enumerate(self._random_seeds):
            np.random.seed(random_seed)
            self._run_experiment(i + 1)
        print('[progress 100.0%] [done]')

    def _run_experiment(self, run_id):
        # Setup associative memory.
        self._memory = AssociativeMemory(map_size=self._map_size,
            map_expanding=self._map_expanding, use_hebbian=self._use_hebbian)
        self._evaluate(run_id, 0)

        # Go through all training rounds.
        for i in range(self._rounds_count):
            first_id   = i * self._categories_new_count
            new_ids    = np.arange(10) + first_id
            retain_ids = np.random.permutation(first_id)[:i
                * self._categories_retain_count]
            select_ids = sorted(np.concatenate([retain_ids, new_ids]) + 1)

            self._run_round(
                run_id,
                i + 1,
                self._stimuli.query('category_id in %r' % select_ids),
            )

    def _run_round(self, run_id, round_id, stimuli):
        # Go through all epochs of this training round.
        for i in range(self._epochs_count):
            self._run_epoch(run_id, round_id, i + 1, stimuli.sample(frac=1))
        self._evaluate(run_id, round_id)

    def _run_epoch(self, run_id, round_id, epoch_id, stimuli):
        # Output current status.
        index    = ((run_id - 1) * self._rounds_count * self._epochs_count
                    + (round_id - 1) * self._epochs_count
                    + (epoch_id - 1))
        count    = len(self._random_seeds) * self._rounds_count * self._epochs_count
        progress = index / count
        print(('[progress %5.1f%%]' % (progress * 100))
            + (' [run %2d/%2d]' % (run_id, len(self._random_seeds)))
            + (' [round %2d/%2d]' % (round_id, self._rounds_count))
            + (' [epoch %2d/%2d]' % (epoch_id, self._epochs_count)))

        # Update dynamic model parameters.
        self._memory.learning_rate = np.interp(
            x=progress,
            xp=(0.0,1.0),
            fp=self._learning_rate_range,
        )
        self._memory.neighborhood_update = np.interp(
            x=progress,
            xp=(0.0,1.0),
            fp=self._neighborhood_update_range
        )
        self._memory.neighborhood_insert = np.interp(
            x=progress,
            xp=(0.0,1.0),
            fp=self._neighborhood_insert_range
        )

        # Train all selected stimuli in this epoch.
        for s in stimuli.itertuples():
            self._memory.train(s.acoustic_vector, s.visual_vector)

    def _evaluate(self, run_id, round_id):
        # Evaluate model based on input categories.
        (
            (conf_a, conf_v),
            (map_a, map_v),
            (count_a, count_v),
            (correct_a, correct_v),
            (gen_a, gen_v),
            taxonomic_factor,
        ) = self._memory.evaluation(
            self._acoustic_categories,
            self._visual_categories,
        )

        # Re-calculate hit map to match `round_id`.
        factor = 1.0 / self._categories_new_count
        map_a = (map_a > 0) * np.floor((map_a - 1) * factor + 1)
        map_v = (map_v > 0) * np.floor((map_v - 1) * factor + 1)

        # Get map sizes.
        size_a1, size_a2 = self._memory.size1
        size_v1, size_v2 = self._memory.size2
        size_a = size_a1 * size_a2
        size_v = size_v1 * size_v2

        # Output evaluation stats.
        print()
        print('Acoustic confusion matrix:')
        print_matrix(np.round(conf_a * 100))
        print()
        print('Visual confusion matrix:')
        print_matrix(np.round(conf_v * 100))
        print()
        print('Acoustic map:')
        print_matrix(map_a)
        print()
        print('Acoustic count:')
        print_matrix(count_a)
        print()
        print('Visual map:')
        print_matrix(map_v)
        print()
        print('Visual count:')
        print_matrix(count_v)
        print()
        print('Acoustic size           : %d (%dx%d)' % (size_a, size_a1, size_a2))
        print('Visual   size           : %d (%dx%d)' % (size_v, size_v1, size_v2))
        print()
        print('Acoustic correct        : %.1f%%' % (correct_a * 100))
        print('Visual   correct        : %.1f%%' % (correct_v * 100))
        print('Acoustic generalization : %.1f%%' % (gen_a * 100))
        print('Visual   generalization : %.1f%%' % (gen_v * 100))
        print('Taxonomic factor        : %.1f%%' % (taxonomic_factor * 100))
        print()

        # Calculate and output error fields for maps.
        err_a = np.linalg.norm(self._memory._map1._weights - self._memory._map1._stimuli, axis=2)
        print('Acoustic errors (%.4f-%.4f, median=%.4f):' % (np.min(err_a), np.max(err_a), np.median(err_a)))
        print_matrix(np.round(err_a / np.max(err_a) * 100))
        print()
        err_v = np.linalg.norm(self._memory._map2._weights - self._memory._map2._stimuli, axis=2)
        print('Visual errors (%.4f-%.4f, median=%.4f):' % (np.min(err_v), np.max(err_v), np.median(err_v)))
        print_matrix(np.round(err_v / np.max(err_v) * 100))
        print()

        # Write to experiment log.
        self._log = self._log.append({
            'experiment_id': self._experiment_id,
            'run_id': run_id,
            'round_id': round_id,
            'map_initial_width': self._map_size[0],
            'map_initial_height': self._map_size[1],
            'map_initial_size': self._map_size[0] * self._map_size[1],
            'map_expanding': self._map_expanding,
            'categories_count': self._categories_count,
            'categories_new_count': self._categories_new_count,
            'categories_retain_count': self._categories_retain_count,
            'variants_count': self._variants_count,
            'epochs_count': self._epochs_count,
            'taxonomic_factor': taxonomic_factor,
            'acoustic_width': size_a1,
            'acoustic_height': size_a2,
            'acoustic_size': size_a,
            'acoustic_correct': correct_a,
            'acoustic_generalization': gen_a,
            'visual_width': size_v1,
            'visual_height': size_v2,
            'visual_size': size_v,
            'visual_correct': correct_v,
            'visual_generalization': gen_v,
            'acoustic_confusion_matrix': array_to_str(conf_a),
            'acoustic_hit_map': array_to_str(map_a),
            'acoustic_hit_count': array_to_str(count_a),
            'acoustic_error_field': array_to_str(err_a),
            'visual_confusion_matrix': array_to_str(conf_v),
            'visual_hit_map': array_to_str(map_v),
            'visual_hit_count': array_to_str(count_v),
            'visual_error_field': array_to_str(err_v),
        }, ignore_index=True)

        # Save experiment log.
        self._log.to_csv(
            experiment_log_path('experiment_%d_log.csv' % self._experiment_id))

        # Output snippet from experiment log.
        print()
        print('---')
        print(self._log)
        print('---')
        print()
