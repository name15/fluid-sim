import numpy as np
from data import Cell

clip = np.frompyfunc(lambda v, lim: -1 if v < 0 or v > lim - 1 else v, 2, 1)

class FluidSim:
    def __init__(self, field: np.ndarray):
        self.field = field
        
    def project(self, iter = 200):
        # Calculate divergence: (left_vx - right_vx + up_vy - down_vy) / 2
        self.field['divergence'][1:-1, 1:-1] = (
            self.field['velocity_x'][1:-1, :-2] - self.field['velocity_x'][1:-1, 2:] + 
            self.field['velocity_y'][:-2, 1:-1] - self.field['velocity_y'][2:, 1:-1]
        ) / 2

        # Calculate pressure: repeat p = (left_p + right_p + down_p + up_p + divergence) / 4
        self.field['pressure'] = np.zeros_like(self.field['pressure']) # clear

        for i in range(iter):            
            self.field['pressure'][1:-1, 1:-1] = ((
                self.field['pressure'][1:-1, :-2] + self.field['pressure'][1:-1, 2:] + 
                self.field['pressure'][:-2, 1:-1] + self.field['pressure'][2:, 1:-1]) + self.field['divergence'][1:-1, 1:-1]
            ) / 4
        
        # Calculate velocity: velocity -= pressure gradient
        self.field['velocity_x'][1:-1, 1:-1] += (
            self.field['pressure'][1:-1, :-2] - self.field['pressure'][1:-1, 2:]
        ) / 2

        self.field['velocity_y'][1:-1, 1:-1] += (
            self.field['pressure'][:-2, 1:-1] - self.field['pressure'][2:, 1:-1]
        ) / 2

        # End result: A vector field with no divergence
    
    def advect(self, dt):
        (ys, xs) = self.field.shape
        (y, x) = np.indices(self.field.shape)
        y = y - self.field['velocity_y'] * dt
        x = x - self.field['velocity_x'] * dt        
        
        yw = y.astype(int)
        xw = x.astype(int)
        yf = y - yw
        xf = x - xw
            
        for key in ['density', 'velocity_x', 'velocity_y']:
            def shift(x, y):
                return self.field[key][clip(yw + y, ys).astype(int), clip(xw + x, xs).astype(int)]

            self.field[key] = \
                (shift(0, 0) * (1 - xf) + shift(1, 0) * xf) * (1 - yf) + \
                (shift(0, 1) * (1 - xf) + shift(1, 1) * xf) * yf
    
    def diffuse(self, iter = 1, k = 0.003):
        # Interpolate values: v = (v + k / 4 * (left_v + right_v + down_v + up_v) ) / (1 + k)
        for i in range(iter):
            for key in ['density', 'velocity_x', 'velocity_y']:
                self.field[key][1:-1, 1:-1] = (
                    self.field[key][1:-1, 1:-1] + (
                        self.field[key][1:-1, 0:-2] + self.field[key][1:-1, 2:] + 
                        self.field[key][0:-2, 1:-1] + self.field[key][2:, 1:-1]
                    ) * k / 4
                ) / (1 + k)