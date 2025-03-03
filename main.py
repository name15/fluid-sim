from data import Field
from viz import FluidViz
from sim import FluidSim
import threading
from time import sleep

# Generate initial field data
grid_size = (480, 720) # (height, width)

def beam(y, x):
    center = (0.5 * grid_size[0], 0.1 * grid_size[1])
    if (y - center[0])**2 + (x - center[1])**2 < 50:
        return 100, 500, 0
    else:
        return 0, 0, 0

field = Field(beam, grid_size)

# Start simulation on this thread
sim = FluidSim(field)

def step():
    sim.project(pressure_iter=10)
    sim.advect(dt=0.5)
    sim.diffuse(iter=1, k=0.03)

# Define state
state = {
    'running': True,
    'paused': True,
    'next': step,
    'show pressure': False,
    'show vectors': True,
    'pen size': 5
}

# Start UI thread
FluidViz.configure_simulation(sim)
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