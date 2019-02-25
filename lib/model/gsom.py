
import numpy as np
from .math_functions import gaussian_field, gaussian_filter

def gsom_example_weights():
    return [
        [   [0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1]   ],
        [   [0,1,0,0],
            [0,1,0,1],
            [0,1,1,0],
            [0,1,1,1]   ],
        [   [1,0,0,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,0,1,1]   ],
        [   [1,1,0,0],
            [1,1,0,1],
            [1,1,1,0],
            [1,1,1,1]   ]
    ]

class GSOM:

    def __init__(self, expanding=True, size=(10, 10), weights=None,
            expand_max_size=640, learning_rate=0.7, expansion_threshold=4.0,
            neighborhood_update=1.0, neighborhood_insert=0.1):
        '''Creates a new Growing Self-Organising Map.

        Setting `expanding` to False prevents the map from growing bigger. In
        this case a fixed `size` should be set.

        It's possible to initialize the map with some specific `weights`, in
        that case the size is determined by the weights themselves and not the
        `size` parameter.


        Parameters
        ----------
        expanding : bool, optional
        size : (int, int), optional
        weights : np.ndarray [shape=(width,height,dpeth)]
        expand_max_size : int
        learning_rate : float
        expansion_threshold : float
        neighborhood_update : float
        neighborhood_insert : float


        Returns
        -------
        GSOM
            new GSOM instance


        Examples
        --------
        >>> GSOM()
        GSOM(size=(10, 10), expanding=True)
        >>> GSOM(size=(2,3))
        GSOM(size=(2, 3), expanding=True)
        >>> GSOM(size=(30,30), expanding=False)
        GSOM(size=(30, 30), expanding=False)
        '''

        # Save parameters.
        self.learning_rate        = learning_rate
        self.expansion_threshold  = expansion_threshold
        self.neighborhood_update  = neighborhood_update
        self.neighborhood_insert  = neighborhood_insert
        self._expanding           = expanding
        self._initial_size        = size
        self._expand_max_size     = expand_max_size
        self._weights             = None
        self._stimuli             = None

        # Initialize weights and error map, if given as parameter.
        if weights is not None:
            self._weights = np.asarray(weights, dtype=np.float64)
            self._stimuli = np.zeros(self._weights.shape)

    @property
    def size(self):
        '''Current size of the map.'''
        if self._weights is not None:
            return self._weights.shape[0:2]
        else:
            return self._initial_size

    def train(self, x):
        '''Train the map to recognize an input stimulus x.

        The training might expand the map, if the BMU error exceeds a threshold
        and the map is allowed to expand.


        Parameters
        ----------
        x : array
            The input stimulus to train, i.e. the map learns this input. The
            input is automatically flattened, which means it can be a vector,
            matrix, or any shape. All inputs must have a fixed number of
            elements / size.


        Returns
        -------
        bmu : (int, int)
            The location on the map of the best-matching unit (BMU)

        direction : (int, int) or None
            The direction that the map expanded or `None` if it didn't expand


        Examples
        --------
        >>> gsom = GSOM(weights=gsom_example_weights())
        >>> gsom.train([1, 0, 1, 1])
        ((2, 3), None)
        '''

        # Flatten the input stimulus and initialize map.
        x = np.ravel(np.asarray(x, dtype=np.float64))
        self._initialize_map(x.size)

        # Find the best-matching unit as well as the BMU quantization error and
        # its threshold for expansion based on previous error.
        bmu, error_field = self._find_bmu(x)
        bmu_error        = error_field[bmu]
        bmu_threshold    = np.linalg.norm(self._weights[bmu] - self._stimuli[bmu]) * self.expansion_threshold

        # If the map is allowed to expand, and we decide the BMU error exceeds
        # the threshold, we expand the map.
        if self._expanding and bmu_error > bmu_threshold:
            # Find out which direction we should expand the map.
            direction = self._find_expand_direction(bmu, error_field)
            # Expand the map in the chosen direction and get new BMU location.
            bmu = self._expand_map(bmu, direction, self.neighborhood_insert)
        else:
            # We don't expand the map, return no direction.
            direction = None

        # Update the map weights to be more similar to the input stimulus.
        self._update_weights(bmu, x, self.neighborhood_update)

        # Return BMU location and expand direction if any.
        return (bmu, direction)

    def activate(self, x):
        # Flatten the input stimulus and initialize map.
        x = np.ravel(np.asarray(x, dtype=np.float64))
        self._initialize_map(x.size)

        # Return activation.
        return self._activate(x)

    def get_bmu_for_activation(self, activation):
        assert activation.ndim == 2
        return np.unravel_index(activation.argmax(), activation.shape)

    def get_bmu_for_stimulus(self, x):
        # Flatten the input stimulus and initialize map.
        x = np.ravel(np.asarray(x, dtype=np.float64))
        self._initialize_map(x.size)

        # Return BMU.
        bmu, error_field = self._find_bmu(x)
        return bmu

    def _initialize_map(self, depth):
        # Don't do anything, if we already initialized the map.
        if self._weights is not None: return

        # Get width/height of map.
        width, height = self._initial_size

        # Create map with initial random weights from a normal distribution.
        # self._weights = np.random.normal(scale=0.01, size=(width, height, depth))
        self._weights = np.random.random(size=(width, height, depth)) * 0.01

        # Initialize the last error of each unit as 0.
        self._stimuli = np.zeros(self._weights.shape)

    def _error_field(self, x):
        # Calculate the error field for the given stimulus x.
        return np.linalg.norm(self._weights - x, axis=2)

    def _activate(self, x):
        return np.exp(-self._error_field(x))

    def _find_bmu(self, x):
        error_field = self._error_field(x)

        # Find the best-matching unit (BMU) location with the minimum error.
        bmu = np.unravel_index(error_field.argmin(), error_field.shape)
        return (bmu, error_field)

    def _update_weights(self, bmu, x, sigma):
        # Get the dimensions of the map.
        width, height, depth = self._weights.shape

        # Calculate the neighborhood function around the BMU. Then update the
        # weights, according to learning rate, neighborhood, and difference
        # between stimulus and current weights.
        g_bmu = gaussian_field(width, height, depth, bmu, sigma)
        self._weights += self.learning_rate * g_bmu * (x - self._weights)

        # Update the last stimuli for the BMU.
        self._stimuli[bmu] = x

    def _find_expand_direction(self, bmu, error_field):
        # Get the (x,y) location of the BMU and width/height of the error field.
        bmu_x, bmu_y  = bmu
        width, height = error_field.shape

        # Get list of candidate directions to expand the map.
        candidates = []
        if bmu_x > 0          : candidates.append((-1,  0))
        if bmu_y > 0          : candidates.append(( 0, -1))
        if bmu_x < width  - 1 : candidates.append(( 1,  0))
        if bmu_y < height - 1 : candidates.append(( 0,  1))

        # Get the errors for each of the candidate directions.
        candidate_locations = np.asarray(candidates) + bmu
        candidate_errors = map(
            lambda location: error_field[tuple(location)],
            candidate_locations)

        # Return the direction with the minimum error.
        return candidates[np.argmin(list(candidate_errors))]

    def _expand_map(self, bmu, direction, sigma):
        # Get the axis to insert (either column or row).
        axis = np.argmax(np.abs(direction))

        # Get the location of the BMU and the neighbor.
        loc_bmu      = bmu[axis]
        loc_neighbor = loc_bmu + direction[axis]

        # Calculate normal distributed random weights for new axis.
        weights_mean = np.mean(self._weights)
        weights_std  = np.std(self._weights)
        weights_new  = np.random.normal(
            loc=weights_mean,
            scale=weights_std,
            size=(self._weights.shape[1 - axis], self._weights.shape[2]),
        )

        # Calculate the average errors of BMU and neighbor column/row.
        stimuli_bmu      = np.take(self._stimuli, loc_bmu, axis)
        stimuli_neighbor = np.take(self._stimuli, loc_neighbor, axis)
        stimuli_avg      = (stimuli_bmu + stimuli_neighbor) * 0.5

        # Insert new column/row using the average weights and errors.
        loc_new       = max(loc_bmu, loc_neighbor)
        self._weights = np.insert(self._weights, loc_new, weights_new, axis)
        self._stimuli = np.insert(self._stimuli, loc_new, stimuli_avg, axis)

        # Update the BMU location to be in the new inserted column/row.
        bmu       = list(bmu)
        bmu[axis] = loc_new
        bmu       = tuple(bmu)

        # Apply a gaussian filter along the inserted axis.
        self._weights += gaussian_filter(self._weights, axis, bmu, sigma)

        # Determine whether we should stop expanding the map from now on.
        width, height, depth = self._weights.shape
        if width * height >= self._expand_max_size:
            self._expanding = False

        # Return new BMU location.
        return bmu

    def __repr__(self):
        return 'GSOM(size=%r, expanding=%r)' % (self.size, self._expanding)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
