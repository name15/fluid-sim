import numpy as np
from data import Cell
from viz import FluidViz
from sim import FluidSim
import threading
from time import sleep

# Generate initial field data
grid_size = (80, 80) # (height, width)

def trig(x, y):
    return np.array((np.sin(x / 7) + np.cos(y / 7), np.cos(x / 7), - np.sin(y / 7), 0, 0), dtype=Cell)

def beam(x, y):
    center = (0.25 * grid_size[1], 0.5 * grid_size[0])
    if (x - center[0])**2 + (y - center[1])**2 < 12:
        return np.array((100, 5000.5, 0, 0, 0), dtype=Cell)
    else:
        return np.array((0, 0, 0, 0, 0), dtype=Cell)

field = np.zeros(grid_size, dtype=Cell)
for y in range(grid_size[0]):
    for x in range(grid_size[1]):
        field[y, x] = beam(x, y)

# Start simulation on this thread
sim = FluidSim(field)

def next_frame():
    print('NEXT FRAME!')
    sim.step()

# Define state
state = {
    'paused': False,
    'next': next_frame
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
    if not state['paused']:
        sim.step()
    else:
        sleep(0.01)