
import numpy as np
from .gsom import GSOM
from .hebbian import Hebbian, FakeHebbian
from .math_functions import distance, gaussian_field

class AssociativeMemory:

    def __init__(self, map_size=(10, 10), map_expanding=True, use_hebbian=True):
        self._map1 = GSOM(size=map_size, expanding=map_expanding)
        self._map2 = GSOM(size=map_size, expanding=map_expanding)

        self._hebbian = FakeHebbian()
        if use_hebbian:
            self._hebbian = Hebbian(
                size1=self._map1.size,
                size2=self._map2.size
            )

    @property
    def size1(self):
        return self._map1.size

    @property
    def size2(self):
        return self._map2.size

    @property
    def learning_rate(self):
        return 0.5 * (self._map1.learning_rate + self._map2.learning_rate)

    @learning_rate.setter
    def learning_rate(self, value):
        self._map1.learning_rate = value
        self._map2.learning_rate = value

    @property
    def neighborhood_update(self):
        return 0.5 * (self._map1.neighborhood_update
            + self._map2.neighborhood_update)

    @neighborhood_update.setter
    def neighborhood_update(self, value):
        self._map1.neighborhood_update = value
        self._map2.neighborhood_update = value

    @property
    def neighborhood_insert(self):
        return 0.5 * (self._map1.neighborhood_insert
            + self._map2.neighborhood_insert)

    @neighborhood_insert.setter
    def neighborhood_insert(self, value):
        self._map1.neighborhood_insert = value
        self._map2.neighborhood_insert = value

    def train(self, stimulus1, stimulus2):
        # Train the two GSOMs with the two stimuli.
        bmu1, direction1 = self._map1.train(stimulus1)
        bmu2, direction2 = self._map2.train(stimulus2)

        # Expand the hebbian connections if necessary.
        direction1 is not None and self._hebbian.expand(bmu1, direction1, index=1)
        direction2 is not None and self._hebbian.expand(bmu2, direction2, index=2)

        # Train the hebbian connections.
        w1, h1 = self._map1.size
        w2, h2 = self._map2.size
        activation1 = gaussian_field(w1, h1, 1, bmu1, 0.5).reshape((w1, h1))
        activation2 = gaussian_field(w2, h2, 1, bmu2, 0.5).reshape((w2, h2))
        self._hebbian.train(activation1, activation2)

    def activate(self, input_stimulus, input_index):
        input_map, output_map = self._get_maps(input_index)

        if isinstance(self._hebbian, FakeHebbian):
            return input_map.get_bmu_for_stimulus(input_stimulus)
        else:
            input_activation  = input_map.activate(input_stimulus)
            output_activation = self._hebbian.activate(
                input_activation, input_index)
            return output_map.get_bmu_for_activation(output_activation)

    def evaluation(self, categories1, categories2):
        # Get list of BMUs and hit map for all categories.
        category1_bmus = self._bmus_for_categories(categories1, 1)
        category2_hits = self._hits_for_categories(categories1, 1)
        category2_bmus = self._bmus_for_categories(categories2, 2)
        category1_hits = self._hits_for_categories(categories2, 2)

        # If not using Hebbian connections, we switch the hit maps to simulate
        # activation on the map itself.
        if isinstance(self._hebbian, FakeHebbian):
            tmp_hits = category1_hits
            category1_hits = category2_hits
            category2_hits = tmp_hits

        # Calculate confusion matrices and generalization factors.
        cm1  = self._confusion_matrix(category1_bmus, category1_hits)
        cm2  = self._confusion_matrix(category2_bmus, category2_hits)
        gen1 = np.sum(np.diag(cm1) >= 0.8) / len(cm1)
        gen2 = np.sum(np.diag(cm2) >= 0.8) / len(cm2)

        return (
            # Confusion matrix.
            (
                cm1,
                cm2,
            ),
            # Hit map.
            (
                self._hit_map(category1_hits, 1),
                self._hit_map(category2_hits, 2),
            ),
            # Hit count.
            (
                self._hit_count(category1_hits, 1),
                self._hit_count(category2_hits, 2),
            ),
            # Average percent correct.
            (
                np.sum(np.diag(cm1)) / len(cm1),
                np.sum(np.diag(cm2)) / len(cm2),
            ),
            # Percent generalization factors.
            (
                gen1,
                gen2,
            ),
            # Taxonomic factor (generalization factors average).
            0.5 * (gen1 + gen2),
        )

    def _bmus_for_categories(self, categories, input_index):
        input_map, output_map = self._get_maps(input_index)
        bmus = [[] for i in range(len(categories))]
        for category_index, stimuli in enumerate(categories):
            for stimulus in stimuli:
                bmu = input_map.get_bmu_for_stimulus(stimulus)
                bmus[category_index].append(bmu)
        return bmus

    def _hits_for_categories(self, categories, input_index):
        hits = dict()
        for category_index, stimuli in enumerate(categories):
            for stimulus in stimuli:
                bmu = self.activate(stimulus, input_index)
                if bmu not in hits:
                    hits[bmu] = np.zeros(len(categories))
                hits[bmu][category_index] += 1
        return hits

    def _confusion_matrix(self, category_bmus, category_hits):
        matrix = []

        for category_index, bmus in enumerate(category_bmus):
            hits = np.zeros(len(category_bmus))
            for bmu in bmus:
                if bmu in category_hits:
                    hits += category_hits[bmu]
                else:
                    distance_hits = [(distance(bmu_hit, bmu), h)
                        for bmu_hit, h in category_hits.items()]
                    min_distance = np.min([d for d, h in distance_hits])
                    distance_hits = np.stack(
                        [h for d, h in distance_hits if d <= min_distance])
                    hits += np.sum(distance_hits, axis=0) / len(distance_hits)
            matrix.append(hits / max(1, np.sum(hits)))

        return np.stack(matrix)

    def _hit_map(self, category_hits, input_index):
        input_map, output_map = self._get_maps(input_index)

        hit_map = np.zeros(input_map.size)
        for bmu, hits in category_hits.items():
            hit_map[bmu] = hits.argmax() + 1
        return hit_map

    def _hit_count(self, category_hits, input_index):
        input_map, output_map = self._get_maps(input_index)

        hit_count = np.zeros(input_map.size)
        for bmu, hits in category_hits.items():
            hit_count[bmu] = np.count_nonzero(hits)
        return hit_count

    def _get_maps(self, input_index):
        maps      = [self._map1, self._map2]
        map_index = input_index - 1
        if isinstance(self._hebbian, FakeHebbian):
            return (maps[map_index], maps[map_index])
        else:
            return (maps[map_index], maps[1 - map_index])
