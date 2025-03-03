import numpy as np
from data import Cell
from viz import FluidViz
from sim import FluidSim
import threading
from time import sleep

# Generate initial field data
grid_size = (180, 360) # (height, width)

def beam(x, y):
    center = (0.1 * grid_size[1], 0.5 * grid_size[0])
    if (x - center[0])**2 + (y - center[1])**2 < 50:
        return np.array((100, 500, 0, 0, 0), dtype=Cell)
    else:
        return np.array((0, 0, 0, 0, 0), dtype=Cell)

field = np.zeros(grid_size, dtype=Cell)
for y in range(grid_size[0]):
    for x in range(grid_size[1]):
        field[y, x] = beam(x, y)

# Start simulation on this thread
sim = FluidSim(field)

def step():
    sim.project(iter=200)
    sim.advect(dt=0.5)
    # sim.diffuse(iter=1, k=0.03)

# Define state
state = {
    'running': True,
    'paused': True,
    'next': step,
    'show pressure': False,
    'show vectors': True
}

# Start UI thread
FluidViz.configure_field(field)
FluidViz.configure_state(state)

def run_viz():
   FluidViz.run()

viz_thread = threading.Thread(target=run_viz)
viz_thread.start()

# Main loop
while True:
    if not state['running']:
        break
    if not state['paused']:
        step()
    else:
        sleep(0.01)