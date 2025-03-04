A very minimal fluid simulation program with projection, (semi-lagrangian) advection and diffusion step. It doesn't even simulate particles (jet), just a grid with densities and velocities.

# Installation and Usage
To install the package:
```bash
git clone https://github.com/name15/fluid-sim.git
pip install -r requirements.txt
```

After installation, you can use:
```bash
python fluid.py -h

usage: python fluid.py [-h] [-p PEN_SIZE] [-s WINDOW_SIZE]
                       [-pi PRESSURE_ITER] [-po PRESSURE_OMEGA]
                       [-at ADVECTION_DT] [-di DIFFUSION_ITER]
                       [-dk DIFFUSION_K]

A very simple fluid simulation program.

options:
  -h, --help            show this help message and exit
  -p PEN_SIZE, --pen_size PEN_SIZE
  -s WINDOW_SIZE, --window_size WINDOW_SIZE
  -pi PRESSURE_ITER, --pressure_iter PRESSURE_ITER
                        Number of iterations for pressure calc. How     
                        curly the fluid is. (default: 20)
  -po PRESSURE_OMEGA, --pressure_omega PRESSURE_OMEGA
                        Omega factor in pressure calc. Equals 1 in      
                        Jakobi iteration and > 1 in successive over-    
                        relaxation. (default: 1.5)
  -at ADVECTION_DT, --advection_dt ADVECTION_DT
  -di DIFFUSION_ITER, --diffusion_iter DIFFUSION_ITER
                        Number of iterations for diffusion. How blured  
                        the fluid is. (default: 1)
  -dk DIFFUSION_K, --diffusion_k DIFFUSION_K
                        Interpolation factor in diffusion. (default:    
                        0.03)

controls:
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
```

![Demo](https://github.com/name15/fluid-sim/blob/71fbee917cf9edd4222cde6940b47b0e7533c333/demo.gif)