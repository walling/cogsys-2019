
import numpy as np

class Hebbian:

    def __init__(self, size1=(10, 10), size2=(10, 10)):
        self._weights = np.zeros(size1 + size2)

    def train(self, activation1, activation2):
        w1, h1, w2, h2 = self._weights.shape
        self._weights += (
            np.asarray(activation1, dtype=np.float64).reshape((w1, h1, 1, 1)) *
            np.asarray(activation2, dtype=np.float64).reshape((1, 1, w2, h2)))

    def expand(self, position, direction, index):
        # Get the axis to insert (either column or row).
        axis = np.argmax(np.abs(direction))

        # Calculate backwards to get the old position.
        position_old = np.copy(position)
        if direction[axis] > 0: position_old[axis] -= 1

        # Calculate the location of the new plane to insert.
        location_new      = position[axis]
        location_current  = position_old[axis]
        location_neighbor = location_current + direction[axis]

        # Calculate the average of the weights, where we insert a new plane.
        w_axis     = (index - 1) * 2 + axis
        w_current  = np.take(self._weights, location_current, w_axis)
        w_neighbor = np.take(self._weights, location_neighbor, w_axis)
        w_avg      = (w_current + w_neighbor) * 0.5

        # Insert new plane using the average weights.
        self._weights = np.insert(self._weights, location_new, w_avg, w_axis)

    def activate(self, input_activation, input_index):
        s1, s2, s3, s4 = self._weights.shape

        if input_index == 1:
            output_activation = np.dot(
                input_activation.reshape(s1 * s2),
                self._weights.reshape((s1 * s2, s3 * s4)),
            ).reshape((s3, s4))
        elif input_index == 2:
            output_activation = np.dot(
                self._weights.reshape((s1 * s2, s3 * s4)),
                input_activation.reshape(s3 * s4),
            ).reshape((s1, s2))
        else:
            raise IndexError('input_index parameter must be either 1 or 2')

        return output_activation

class FakeHebbian:

    def __init__(self): pass

    def train(self, activation1, activation2): pass

    def expand(self, position, direction, index): pass

    def activate(self, input_activation, input_index):
        return input_activation
