import moderngl
import moderngl_window as mglw 
from data import Field2D
import numpy as np

# Define programs for rendering grid and arrows

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
arrow_vertex_shader = """
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
arrow_fragment_shader = """
#version 330
out vec4 f_color;
void main(){
    f_color = vec4(0.0, 1.0, 1.0, 1.0);
}
"""

def update_arrows(arrows: np.ndarray, grid_size: tuple, velocities_x: np.ndarray, velocities_y: np.ndarray):
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            i = (y * grid_size[0] + x) * 4
            
            start_x = x + 0.5
            start_y = y + 0.5

            dx = velocities_x[x, y]
            dy = velocities_y[x, y]
            
            end_x = start_x + dx
            end_y = start_y + dy
            
            arrows[i]   = start_x
            arrows[i+1] = start_y
            arrows[i+2] = end_x
            arrows[i+3] = end_y


def update_colors(colors: np.ndarray, grid_size: tuple, density: np.ndarray):
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            i = (y * grid_size[0] + x) * 12
            c = density[x, y] + 0.1
            for j in range(12):
                colors[i + j] = c


class FluidViz(mglw.WindowConfig):
    title = "Fluid demo"
    window_size = (800, 600)

    @classmethod
    def configure(cls, field: Field2D):
        cls.field = field
        cls.grid_size = field.size

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Generate mesh (just once)
        gs = self.grid_size
        vertices = []
        faces = []
        d = 0.0 # Grid spacing
        for y in range(gs[1]):
            for x in range(gs[0]):
                bl = (x + d, y + d)
                br = (x + 1 - d, y + d)
                tl = (x + d, y + 1 - d)
                tr = (x + 1 - d, y + 1 - d)
                vertices.extend([bl, br, tr, tl])
                l = len(vertices)
                faces.extend([l-4, l-2, l-3, l-4, l-1, l-2])
        
        self.vertices = np.array(vertices, dtype='f4')
        
        self.faces = np.array(faces, dtype='i4')
        self.colors = np.zeros(gs[0] * gs[1] * 12, dtype='f4')
        
        # Create buffers for mesh
        self.mesh_vbo = self.ctx.buffer(self.vertices.tobytes())
        self.mesh_ibo = self.ctx.buffer(self.faces.tobytes())

        # Compile shader program for mesh
        self.grid_prog = self.ctx.program(vertex_shader=grid_vertex_shader, fragment_shader=grid_fragment_shader)
        self.keys = {}
        self.translate = np.array((gs[0] / 2, gs[1] / 2), dtype='f4')
        self.grid_prog['translate'].write(self.translate.tobytes())
        self.scale = np.array(min(2/gs[0], 2/gs[1]), dtype='f4')
        self.grid_prog['scale'].write(self.scale.tobytes())
        
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

        # Setup arrow buffer and program (dynamic buffer updated per frame)
        self.arrows = np.empty(gs[0] * gs[1] * 4, dtype=np.float32)
        self.arrow_vbo = self.ctx.buffer(self.arrows.tobytes(), dynamic=True)
        self.arrow_prog = self.ctx.program(vertex_shader=arrow_vertex_shader, fragment_shader=arrow_fragment_shader)
        self.arrow_vao = self.ctx.vertex_array(
            self.arrow_prog,
            [(self.arrow_vbo, '2f', 'in_position')]
        )
    
    def on_render(self, time, frame_time):
        # Update viewport and ratio
        width, height = self.wnd.size
        self.ctx.viewport = (0, 0, width, height)
        self.ratio = np.array(width / height, dtype='f4').tobytes()
        
        # Update transformation uniforms
        self.transform()

        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.0)

        # Update arrow positions buffer
        update_colors(self.colors, self.grid_size, self.field.data['density'])
        self.color_vbo.write(self.colors.tobytes())

        # Update arrow positions buffer
        update_arrows(self.arrows, self.grid_size, self.field.data['velocity_x'], self.field.data['velocity_y'])
        self.arrow_vbo.write(self.arrows.tobytes())

        # Draw mesh and arrows
        self.grid_vao.render(mode=moderngl.TRIANGLES)
        self.arrow_vao.render(mode=moderngl.LINES)

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
        if self.keys.get(self.wnd.keys.Q, False):
            self.scale *= 1.05
        if self.keys.get(self.wnd.keys.E, False):
            self.scale /= 1.05
        
        move /= np.linalg.norm(move) + 0.01
        self.translate += move

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            self.keys[key] = True
        if action == self.wnd.keys.ACTION_RELEASE:
            self.keys[key] = False
    
    def transform(self):
        # TODO: Use mvp instead
        self.grid_prog['translate'].write(self.translate.tobytes())
        self.grid_prog['scale'].write(self.scale.tobytes())
        self.grid_prog['ratio'].write(self.ratio)
        self.arrow_prog['translate'].write(self.translate.tobytes())
        self.arrow_prog['scale'].write(self.scale.tobytes())
        self.arrow_prog['ratio'].write(self.ratio)
    