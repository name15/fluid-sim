import numpy as np

Cell = np.dtype([
    ('density', np.float32),
    ('velocity_x', np.float32),
    ('velocity_y', np.float32),
    ('pressure', np.float32),
    ('divergence', np.float32)
])