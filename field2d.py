import numpy as np

Cell = np.dtype([
    ('velocity_x', np.float32),
    ('velocity_y', np.float32),
    ('density', np.float32),
])

class Field2D:
    def __init__(self, size: tuple[int, int], generator):
        self.size = np.array(size)

        self.data = np.zeros(size, dtype=Cell)
        for x in range(size[0]):
            for y in range(size[1]):
                self.data[x, y] = generator(x, y)
            
    def __getitem__(self, coords):
        return self.data[coords]

    