
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys
import numpy as np
from . import load_all_logs
from ..file import experiment_graph_path

class ExperimentAnalyzer:

    def __init__(self):
        self._log = load_all_logs()

    def run(self):
        # Output log summary.
        log = self._log
        print('---')
        print(log)
        print('---')
        print(log.describe())
        print('---')
        print()

        # Analyze all rounds in all experiments.
        experiments = {}
        for (eid, rid), data in log.groupby(['experiment_id', 'round_id']):
            if eid not in experiments: experiments[eid] = []
            experiments[eid].append(self._run_round(eid, rid, data))

        for eid, data in experiments.items():
            self._run_experiment(eid, np.stack(data))

        for eid, data in experiments.items():
            self._run_experiment_summary(eid, np.stack(data))

    def _run_round(self, eid, rid, data):
        print('[experiment%2d] [round%2d]' % (eid, rid))

        # Get various properties.
        i_map  = min(3, data.categories_count.count() - 1)
        n_cat  = data.categories_count.iloc[0]
        w_a    = data.acoustic_width.iloc[i_map]
        h_a    = data.acoustic_height.iloc[i_map]
        w_v    = data.visual_width.iloc[i_map]
        h_v    = data.visual_height.iloc[i_map]
        size_c = (n_cat, n_cat)
        size_a = (w_a, h_a)
        size_v = (w_v, h_v)

        # Confusion matrices.
        conf_a = np.mean(
            np.stack(data.acoustic_confusion_matrix),
            axis=0,
        ).reshape(size_c)
        conf_v = np.mean(
            np.stack(data.visual_confusion_matrix),
            axis=0,
        ).reshape(size_c)
        conf_a_std = np.std(
            np.stack(data.acoustic_confusion_matrix),
            axis=0,
        ).reshape(size_c)
        conf_v_std = np.std(
            np.stack(data.visual_confusion_matrix),
            axis=0,
        ).reshape(size_c)

        # Render confusion matrices.
        self._conf_matrix(eid, rid, cm=conf_a, label='acoustic')
        self._conf_matrix(eid, rid, cm=conf_v, label='visual')
        self._conf_matrix(eid, rid, cm=conf_a_std, label='acoustic_std', normalize=False, cmap=plt.cm.Reds)
        self._conf_matrix(eid, rid, cm=conf_v_std, label='visual_std', normalize=False, cmap=plt.cm.Reds)

        # Error fields.
        err_a = data.acoustic_error_field.iloc[i_map].reshape(size_a)
        err_v = data.visual_error_field.iloc[i_map].reshape(size_v)

        # Render error fields.
        self._map(eid, rid, err_a, label='error_field_acoustic', cmap=plt.cm.Reds)
        self._map(eid, rid, err_v, label='error_field_visual', cmap=plt.cm.Reds)

        # Hit counts.
        count_a = data.acoustic_hit_count.iloc[i_map].reshape(size_a)
        count_v = data.visual_hit_count.iloc[i_map].reshape(size_v)

        # Render hit counts.
        self._map(eid, rid, count_a, label='hit_count_acoustic', cmap=plt.cm.Greys)
        self._map(eid, rid, count_v, label='hit_count_visual', cmap=plt.cm.Greys)

        # Hit maps.
        map_a = data.acoustic_hit_map.iloc[i_map].reshape(size_a)
        map_v = data.visual_hit_map.iloc[i_map].reshape(size_v)

        # Render hit maps.
        cmap_hit = LinearSegmentedColormap.from_list(
            'cmap_hit',
            colors=[
                (255/255, 255/255, 255/255),
                (221/255, 104/255,  26/255),
                ( 26/255, 143/255, 216/255),
                (159/255, 224/255,  40/255),
            ],
            N=4,
        )
        self._map(eid, rid, map_a, label='hit_map_acoustic', cmap=cmap_hit)
        self._map(eid, rid, map_v, label='hit_map_visual', cmap=cmap_hit)

        # Aggregate numeric results.
        return np.array([
            np.mean(np.stack(data.taxonomic_factor)),
            np.std(np.stack(data.taxonomic_factor)),
            np.mean(np.stack(data.acoustic_generalization)),
            np.std(np.stack(data.acoustic_generalization)),
            np.mean(np.stack(data.visual_generalization)),
            np.std(np.stack(data.visual_generalization)),
            np.mean(np.stack(data.acoustic_correct)),
            np.std(np.stack(data.acoustic_correct)),
            np.mean(np.stack(data.visual_correct)),
            np.std(np.stack(data.visual_correct)),
            np.mean(np.stack(data.acoustic_size)),
            np.std(np.stack(data.acoustic_size)),
            np.mean(np.stack(data.visual_size)),
            np.std(np.stack(data.visual_size)),
        ])

    def _run_experiment(self, eid, data):
        print('[experiment%2d] [all]' % eid)

        fig_filename = experiment_graph_path('e%02d_taxonomic_bar.pdf' % eid)

        ys = np.arange(0, 101, 10)
        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.bar([0,1], [
            data[-1, 2]*100,
            data[-1, 4]*100,
        ], yerr=[
            data[-1, 3]*100,
            data[-1, 5]*100,
        ], tick_label=[
            'Comprehension',
            'Production',
        ], color=['c', 'g'])
        plt.yticks(ys, ys)
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')

        fig_filename = experiment_graph_path('e%02d_taxonomic_graph.pdf' % eid)

        xs = np.arange(len(data))
        ys = np.arange(0, 101, 25)
        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.errorbar(xs, data[:, 0]*100, yerr=data[:, 1]*100, fmt='ko-')
        plt.plot(xs, data[:, 2]*100, 'cx:')
        plt.plot(xs, data[:, 4]*100, 'gx:')
        plt.xticks(xs, xs)
        plt.yticks(ys, ys)
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')

        fig_filename = experiment_graph_path('e%02d_correct.pdf' % eid)

        xs = np.arange(len(data))
        ys = np.arange(0, 101, 25)
        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.errorbar(xs, data[:, 6]*100, yerr=data[:, 7]*100, fmt='co--')
        plt.errorbar(xs, data[:, 8]*100, yerr=data[:, 9]*100, fmt='go--')
        plt.xticks(xs, xs)
        plt.yticks(ys, ys)
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')

        fig_filename = experiment_graph_path('e%02d_map_size.pdf' % eid)

        xs = np.arange(len(data))
        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.errorbar(xs, data[:, 10], yerr=data[:, 11], fmt='cD-')
        plt.errorbar(xs, data[:, 12], yerr=data[:, 13], fmt='gD-')
        plt.xticks(xs, xs)
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')

    def _run_experiment_summary(self, eid, data):
        print()
        print('Experiment %d:' % eid)
        print()

        latex = '%d' % eid
        latex += '\n    & $%.0f \\pm %.1f$' % (data[-1, 12], data[-1, 13])
        latex += '\n    & $%.0f \\pm %.1f$' % (data[-1, 10], data[-1, 11])

        if eid <= 2:
            latex += '\n    & $%.1f\\%% \\pm %.1f$' % (data[-1, 8]*100, data[-1, 9]*100)
            latex += '\n    & $%.1f\\%% \\pm %.1f$' % (data[-1, 6]*100, data[-1, 7]*100)
        else:
            latex += '\n    & n/a'
            latex += '\n    & n/a'

        if eid >= 3:
            latex += '\n    & $%.1f\\%% \\pm %.1f$' % (data[-1, 2]*100, data[-1, 3]*100)
            latex += '\n    & $%.1f\\%% \\pm %.1f$' % (data[-1, 4]*100, data[-1, 5]*100)
            latex += '\n    & $%.1f\\%% \\pm %.1f$' % (data[-1, 0]*100, data[-1, 1]*100)
        else:
            latex += '\n    & n/a'
            latex += '\n    & n/a'
            latex += '\n    & n/a'

        latex += '\n    \\\\'

        print(latex)
        print()

    def _conf_matrix(self, eid, rid, cm, label, normalize=True, cmap=plt.cm.Blues):
        if normalize: cm /= cm.sum(axis=1)[:, np.newaxis]

        fig_filename = experiment_graph_path(
            'e%02d_confusion_matrix_%s_r%02d.pdf' % (eid, label, rid))

        xs = np.array([1, 10, 20, 30])
        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.xticks(xs-1, xs)
        plt.yticks(xs-1, xs)
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')

    def _map(self, eid, rid, im, label, cmap=plt.cm.Reds):
        if im.shape[0] > im.shape[1]: im = im.T

        fig_filename = experiment_graph_path(
            'e%02d_%s_r%02d.pdf' % (eid, label, rid))

        plt.ioff()
        plt.figure(figsize=(7, 7))
        plt.imshow(im, interpolation='nearest', cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
        plt.close('all')
