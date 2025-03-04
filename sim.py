from data import Field
import numpy as np
import numba as nb

@nb.njit(cache=True)
def calc_divergence(divergence, velocity_x, velocity_y):
    # Calculate divergence
    # (left_vx - right_vx + up_vy - down_vy) / 2
    divergence[1:-1, 1:-1] = (
        velocity_x[1:-1, :-2] - velocity_x[1:-1, 2:] + 
        velocity_y[:-2, 1:-1] - velocity_y[2:, 1:-1]
    ) / 2

@nb.njit(parallel=True, cache=True)
def calc_pressure(pressure, divergence, iter, omega):
    # Calculate pressure using jakobi iteration
    pressure.fill(0) # clear

    for i in range(iter):
        for y in nb.prange(1, pressure.shape[0]-1):
            for x in nb.prange(1, pressure.shape[1]-1):
                # p = (left_p + right_p + down_p + up_p + divergence) / 4
                pressure_new = (
                    pressure[y, x-1] + pressure[y, x+1] + 
                    pressure[y-1, x] + pressure[y+1, x] + 
                    divergence[y, x]
                ) / 4.0
                
                # successive overrelaxation 
                pressure[y, x] = pressure[y, x] + omega * (pressure_new - pressure[y, x])

@nb.njit(cache=True)
def apply_pressure(pressure, velocity_y, velocity_x):
    # Apply pressure gradient
    # velocity -= pressure gradient
    
    # The result is a divergence-free vector field
    velocity_x[1:-1, 1:-1] += (
        pressure[1:-1, :-2] - pressure[1:-1, 2:]
    ) / 2

    velocity_y[1:-1, 1:-1] += (
        pressure[:-2, 1:-1] - pressure[2:, 1:-1]
    ) / 2

@nb.njit(parallel=True, cache=True)
def advect(density, velocity_y, velocity_x, new_density, new_velocity_y, new_velocity_x, dt):
    height, width = density.shape

    new_density.fill(0)
    new_velocity_x.fill(0)
    new_velocity_y.fill(0)

    for y in nb.prange(height-1):
        for x in nb.prange(width-1):
            # Backtrace coordinates
            py = float(y) - velocity_y[y, x] * dt
            px = float(x) - velocity_x[y, x] * dt
            
            # Whole part
            y0 = int(py)
            x0 = int(px)
            y1 = y0 + 1
            x1 = x0 + 1

            if y0 < 0 or y0 >= height or x0 < 0 or x0 > width \
            or y1 < 0 or y1 >= height or x1 < 0 or x1 > width:
                continue

            # Fractional part
            fy = py - y0
            fx = px - x0

            # Bilinear interpolation
            new_density[y, x] = (density[y0, x0] * (1 - fx) + density[y0, x1] * fx) * (1 - fy) + \
                (density[y1, x0] * (1 - fx) + density[y1, x1] * fx) * fy

            new_velocity_y[y, x] = (velocity_y[y0, x0] * (1 - fx) + velocity_y[y0, x1] * fx) * (1 - fy) + \
                (velocity_y[y1, x0] * (1 - fx) + velocity_y[y1, x1] * fx) * fy
        
            new_velocity_x[y, x] = (velocity_x[y0, x0] * (1 - fx) + velocity_x[y0, x1] * fx) * (1 - fy) + \
                (velocity_x[y1, x0] * (1 - fx) + velocity_x[y1, x1] * fx) * fy    


@nb.njit(cache=True)
def diffuse(density, velocity_y, velocity_x, new_density, new_velocity_y, new_velocity_x, iter, k):
    # Interpolate values
    # v = (v + k / 4 * (left_v + right_v + down_v + up_v) ) / (1 + k)
    def interpolate(field, y, x, k):
        return (field[y, x] + (field[y + 1, x] + field[y - 1, x] + field[y, x + 1] + field[y, x - 1]) * k / 4) / (1.0 + k)

    for i in range(iter):
        for y in nb.prange(1, density.shape[0] - 1):
            for x in nb.prange(1, density.shape[1] - 1):
                new_density[y, x] = interpolate(density, y, x, k)
                new_velocity_y[y, x] = interpolate(velocity_y, y, x, k)
                new_velocity_x[y, x] = interpolate(velocity_x, y, x, k)
        
        density, new_density = new_density, density
        velocity_x, new_velocity_x = new_velocity_x, velocity_x
        velocity_y, new_velocity_y = new_velocity_y, velocity_y

class FluidSim:
    def __init__(self, field: np.ndarray):
        self.front = field
        self.back = Field(lambda y, x: field[y, x], field.shape)        

    def project(self, pressure_iter = 100, omega = 1.5):
        calc_divergence(self.front.divergence, self.front.velocity_x, self.front.velocity_y)
        calc_pressure(self.front.pressure, self.front.divergence, pressure_iter, omega)        
        apply_pressure(self.front.pressure, self.front.velocity_y, self.front.velocity_x)
    
    def advect(self, dt):
        advect(self.front.density, self.front.velocity_y, self.front.velocity_x, self.back.density, self.back.velocity_y, self.back.velocity_x, dt)
        self.front, self.back = self.back, self.front
        
    def diffuse(self, iter = 1, k = 0.003):
        diffuse(self.front.density, self.front.velocity_y, self.front.velocity_x, self.back.density, self.back.velocity_y, self.back.velocity_x, iter, k)
        if iter % 2 == 1:
            self.front, self.back = self.back, self.front
        
    @classmethod
    def warm_up(cls):
        test = np.zeros((4, 4), dtype=np.float32)
        print('- Kernel: calc_divergence')
        calc_divergence(test, test, test)
        print('- Kernel: calc_pressure')
        calc_pressure(test, test, 1, 1)
        print('- Kernel: apply_pressure')
        apply_pressure(test, test, test)
        print('- Kernel: advect')
        advect(test, test, test, test, test, test, 1)
        print('- Kernel: diffuse')
        diffuse(test, test, test, test, test, test, 1, 1)