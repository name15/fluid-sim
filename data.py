import numpy as np

Cell = np.dtype([
    ('velocity_x', np.float32),
    ('velocity_y', np.float32),
    ('density', np.float32),
])

class Field2D:
    def __init__(self, size: tuple[int, int], generator):
        self.size = np.array(size)

        self.data = np.zeros((size[1], size[0]), dtype=Cell)
        for y in range(size[1]):
            for x in range(size[0]):
                self.data[y, x] = generator(x, y)
    