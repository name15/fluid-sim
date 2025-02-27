import numpy as np
from data import Cell, Field2D
from viz import FluidViz
from sim import FluidSim
import threading

# Generate initial field data
grid_size = (80, 60)

def generator(x, y):
    center = (0.25 * grid_size[0], 0.5 * grid_size[1])
    if (x - center[0])**2 + (y - center[1])**2 < 10:
        return np.array((1000, 0, 1), dtype=Cell)
    else:
        return np.array((0, 0, 0), dtype=Cell)

field = Field2D(grid_size, generator) # Shared across threads 

# Start UI thread
FluidViz.configure(field)

def run_viz():
   FluidViz.run()

viz_thread = threading.Thread(target=run_viz)
viz_thread.start()

# Start simulation on this thread
sim = FluidSim(field)

# Main loop
frame = 0

while True:
    sim.step(0.01)

    if frame % 100 == 0 and not viz_thread.is_alive():
        break
    frame += 1