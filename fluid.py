from data import Field
from viz import FluidViz
from sim import FluidSim
import threading
from time import sleep
from argparse import ArgumentParser, HelpFormatter, SUPPRESS, OPTIONAL, ZERO_OR_MORE

class Formatter(HelpFormatter):
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not SUPPRESS:
                defaulting_nargs = [OPTIONAL, ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help

parser = ArgumentParser(
    prog='python fluid.py',
    description="A very simple fluid simulation program.",
    epilog="""controls:
  Mouse drag: Add density and alter velocity under the cursor 
  P: Pause/Unpause
  W: Up
  A: Down
  S: Left
  D: Right
  E: Zoom out
  Q: Zoom in 
  R: Reset view
  P: Show/Hide pressure
  V: Show/Hide vectors
  C: Scale vectors down
  B: Scale vectors up
""",
    formatter_class=Formatter
)

parser.add_argument('-p', '--pen_size', default=5)
parser.add_argument('-s', '--window_size', default='720x480')
parser.add_argument('-pi', '--pressure_iter', default=20, help='Number of iterations for pressure calc. How curly the fluid is.')
parser.add_argument('-po', '--pressure_omega', default=1.5, help='Omega factor in pressure calc. Equals 1 in Jakobi iteration and > 1 in successive over-relaxation.')
parser.add_argument('-at', '--advection_dt', default=0.5)
parser.add_argument('-di', '--diffusion_iter', default=1, help='Number of iterations for diffusion. How blured the fluid is.')
parser.add_argument('-dk', '--diffusion_k', default=0.03, help='Interpolation factor in diffusion.')

args = parser.parse_args()

# Generate initial field data
s = args.window_size.split('x')
grid_size = (int(s[1]), int(s[0])) # (height, width)

# Compiile shaders
print("Compiling kernels...")
FluidSim.warm_up()
print("Starting simulation...")

def beam(y, x):
    center = (0.5 * grid_size[0], 0.1 * grid_size[1])
    if (y - center[0])**2 + (x - center[1])**2 < (grid_size[0]/10)**2:
        return 1000, 5000, 0
    else:
        return 0, 0, 0

field = Field(beam, grid_size)

# Start simulation on this thread
sim = FluidSim(field)

def step():
    sim.project(args.pressure_iter, args.pressure_omega)
    sim.advect(args.advection_dt)
    sim.diffuse(args.diffusion_iter, args.diffusion_k)

field = Field(beam, grid_size)

# Define state
state = {
    'running': True,
    'paused': False,
    'next': step,
    'show pressure': False,
    'show vectors': False,
    'pen size': args.pen_size
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