import moderngl
import moderngl_window as mglw 
import numpy as np

# Define programs for rendering grid and vectors

grid_vertex_shader = """
#version 330
in vec2 in_position;
in vec3 in_color;
out vec3 v_color;
uniform vec2 translate;
uniform float ratio;
uniform float scale;

void main(){
    v_color = in_color;
    vec2 pos = in_position - translate;
    pos.x /= ratio;
    pos *= scale;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""
grid_fragment_shader = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main(){
    f_color = vec4(v_color, 1.0);
}
"""
vector_vertex_shader = """
#version 330
in vec2 in_position;
uniform vec2 translate;
uniform float ratio;
uniform float scale;

void main() {
    vec2 pos = in_position - translate;
    pos.x /= ratio;
    pos *= scale;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""
vector_fragment_shader = """
#version 330
out vec4 f_color;
void main(){
    f_color = vec4(0.0, 0.62, 0.62, 1.0);
}
"""


class FluidViz(mglw.WindowConfig):
    title = "Fluid Sim"
    window_size = (800, 600)

    @classmethod
    def configure_simulation(cls, simulation):
        cls.simulation = simulation

    @classmethod
    def configure_state(cls, state):
        cls.state = state

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Generate mesh
        gs = self.simulation.front.shape
        (y, x) = np.indices(gs)
        (yf, xf) = (y.flatten(), x.flatten())
        self.vertices = np.empty((gs[1] * gs[0] * 4, 2), dtype='f4')
        self.vertices[0::4] = np.stack([xf, yf], axis=1)
        self.vertices[1::4] = np.stack([xf + 1, yf], axis=1)
        self.vertices[2::4] = np.stack([xf + 1, yf + 1], axis=1)
        self.vertices[3::4] = np.stack([xf, yf + 1], axis=1)
        
        self.faces = np.empty((gs[1] * gs[0] * 6), dtype='i4')
        l = 4 * np.arange(gs[0] * gs[1])
        self.faces[0::6] = l
        self.faces[1::6] = l + 2
        self.faces[2::6] = l + 1
        self.faces[3::6] = l
        self.faces[4::6] = l + 3
        self.faces[5::6] = l + 2

        # Init color array
        self.colors = np.zeros((gs[0], gs[1], 12), dtype='f4')
        
        # Create buffers for mesh
        self.mesh_vbo = self.ctx.buffer(self.vertices.tobytes())
        self.mesh_ibo = self.ctx.buffer(self.faces.tobytes())

        # Compile shader program for mesh
        self.grid_prog = self.ctx.program(vertex_shader=grid_vertex_shader, fragment_shader=grid_fragment_shader)
        self.keys = {}
        
        # Create a buffer for vertex colors
        self.color_vbo = self.ctx.buffer(self.colors.tobytes(), dynamic=True)

        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [
            (self.mesh_vbo, '2f', 'in_position'),
            (self.color_vbo, '3f', 'in_color')
            ],
            self.mesh_ibo
        )

        # Setup vector buffer and program (dynamic buffer updated per frame)
        self.vectors = np.empty((gs[0], gs[1], 4), dtype=np.float32)
        self.vectors[:, :, 0] = x + 0.5 # Start x
        self.vectors[:, :, 1] = y + 0.5 # Start y
        
        self.vector_vbo = self.ctx.buffer(self.vectors.tobytes(), dynamic=True)
        self.vector_prog = self.ctx.program(vertex_shader=vector_vertex_shader, fragment_shader=vector_fragment_shader)
        self.vector_vao = self.ctx.vertex_array(
            self.vector_prog,
            [(self.vector_vbo, '2f', 'in_position')]
        )

        # Initialize viewing variables
        self.center()
        self.vector_scale = 1
        self.last_pos = None
    
    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_RELEASE:
            self.keys[key] = False
        if action == self.wnd.keys.ACTION_PRESS:
            self.keys[key] = True
            if key == self.wnd.keys.SPACE:
                self.state['paused'] = not self.state['paused']
            if key == self.wnd.keys.RIGHT:
                self.state['next']()
            if key == self.wnd.keys.P:
                self.state['show pressure'] = not self.state['show pressure']
            if key == self.wnd.keys.V:
                self.state['show vectors'] = not self.state['show vectors']        
    
    def on_render(self, time, frame_time):
        # Update viewport and ratio
        width, height = self.wnd.size
        self.ctx.viewport = (0, 0, width, height)
        self.ratio = np.array(width / height, dtype='f4')
        
        # Update transformation uniforms
        self.transform()

        # Clear screen
        self.ctx.clear(0.078, 0.137, 0.169)

        # Update vector positions buffer
        self.colors[:, :, :] = self.simulation.front.density[:, :, np.newaxis]
        if self.state['show pressure']:
            self.colors[:, :, 0::3] = - self.simulation.front.pressure[:, :, np.newaxis]
            self.colors[:, :, 2::3] = self.simulation.front.pressure[:, :, np.newaxis]
        self.color_vbo.write(self.colors.tobytes())

        # Update vector positions buffer
        self.vectors[:, :, 2] = self.vectors[:, :, 0]
        self.vectors[:, :, 3] = self.vectors[:, :, 1]
        if self.state['show vectors']:
            self.vectors[:, :, 2] += self.vector_scale * self.simulation.front.velocity_x # End x
            self.vectors[:, :, 3] += self.vector_scale * self.simulation.front.velocity_y # End y
        self.vector_vbo.write(self.vectors.tobytes())

        # Draw mesh and vectors
        self.grid_vao.render(mode=moderngl.TRIANGLES)
        self.vector_vao.render(mode=moderngl.LINES)

        # Process user input
        move = np.array([0, 0], dtype='f4')
        if self.keys.get(self.wnd.keys.A, False):
            move[0] -= 0.1
        if self.keys.get(self.wnd.keys.D, False):
            move[0] += 0.1
        if self.keys.get(self.wnd.keys.S, False):
            move[1] -= 0.1
        if self.keys.get(self.wnd.keys.W, False):
            move[1] += 0.1
        if self.keys.get(self.wnd.keys.E, False):
            self.scale /= 1.05
        if self.keys.get(self.wnd.keys.Q, False):
            self.scale *= 1.05
        if self.keys.get(self.wnd.keys.R, False):
            self.center()
        if self.keys.get(self.wnd.keys.C, False):
            self.vector_scale /= 1.05
        if self.keys.get(self.wnd.keys.B, False):
            self.vector_scale *= 1.05
        
        
        move /= np.linalg.norm(move) + 0.01
        self.translate += 0.075 * move / self.scale
    
    def on_mouse_drag_event(self, x, y, dx, dy):
        # print(x, y, dx, dy)
        width, height = self.wnd.size
        ys, xs = self.simulation.front.shape
        ps = self.state['pen size']
        y, x = 1 - y / height * 2, x / width * 2 - 1
        ya = y / self.scale.item() + self.translate[1]
        xa = x / self.scale.item() * self.ratio.item() + self.translate[0]
        
        if self.last_pos:
            yb, xb = self.last_pos[0], self.last_pos[1]
            for i in range(0, 10):
                t = i / 10
                yt, xt = ya * (1 - t) + yb * t, xa * (1 - t) + xb * t  
                for dx in range(-ps, ps):
                    for dy in range(-ps, ps):
                        if dx**2 + dy**2 < ps**2:
                            yn = yt + dy
                            xn = xt + dx
                            if yn < 0 or xn < 0 or yn >= ys or xn >= xs:
                                continue
                            self.simulation.front[int(yn), int(xn)] = 1, xa - xb, ya - yb
        
        self.last_pos = (ya, xa)
    
    def on_mouse_release_event(self, x, y, button):
        self.last_pos = None

    def center(self):
        gs = self.simulation.front.shape
        self.translate = np.array((gs[1] / 2, gs[0] / 2), dtype='f4')
        self.scale = np.array(min(2/gs[1], 2/gs[0]), dtype='f4')

    def transform(self):
        # TODO: Use mvp instead
        self.grid_prog['translate'].write(self.translate.tobytes())
        self.grid_prog['scale'].write(self.scale.tobytes())
        self.grid_prog['ratio'].write(self.ratio.tobytes())
        self.vector_prog['translate'].write(self.translate.tobytes())
        self.vector_prog['scale'].write(self.scale.tobytes())
        self.vector_prog['ratio'].write(self.ratio.tobytes())

    def on_close(self):
        super().on_close()
        self.state['running'] = False