import numpy as np

class Field:
    def __init__(self, generator, shape):
        self.shape = shape
        self.density = np.zeros(shape, dtype=np.float32)
        self.velocity_x = np.zeros(shape, dtype=np.float32)
        self.velocity_y = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.divergence = np.zeros(shape, dtype=np.float32)

        for y in range(shape[0]):
            for x in range(shape[1]):
                self[y, x] = generator(y, x)
    
    def __getitem__(self, key):
        return self.density[key], self.velocity_x[key], self.velocity_y[key]

    def __setitem__(self, key, value):
        self.density[key] = value[0]
        self.velocity_x[key] = value[1]
        self.velocity_y[key] = value[2]