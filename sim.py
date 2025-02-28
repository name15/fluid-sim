import numpy as np
from data import Cell

clip = np.frompyfunc(lambda a, b: -1 if a > b - 1 else int(a), 2, 1)

class FluidSim:
    def __init__(self, field: np.ndarray):
        self.field = field
    
    def step(self, dt = 0.01):
        self.project()
        self.advect(dt)
    
    def project(self, iterations = 100):
        # Calculate divergence: (left_vx - right_vx + up_vy - down_vy) / 2
        self.field['divergence'][1:-1, 1:-1] = (
            self.field['velocity_x'][1:-1, :-2] - self.field['velocity_x'][1:-1, 2:] + 
            self.field['velocity_y'][:-2, 1:-1] - self.field['velocity_y'][2:, 1:-1]
        ) / 2

        # Calculate pressure: repeat p = (left_p + right_p + down_p + up_p + divergence) / 4
        self.field['pressure'] = np.zeros_like(self.field['pressure']) # clear

        for i in range(iterations):            
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

        dl = self.field[yw, xw][:]
        dr = self.field[yw, clip(xw + 1, xs).astype(int)][:]
        tl = self.field[clip(yw + 1, ys).astype(int), xw][:]
        tr = self.field[clip(yw + 1, ys).astype(int), clip(xw + 1, xs).astype(int)][:]
        
        for key in ['density', 'velocity_x', 'velocity_y']:
            self.field[key] = \
                (dl[key] * (1 - xf) + dr[key] * xf) * (1 - yf) + \
                (tl[key] * (1 - xf) + tr[key] * xf) * yf
