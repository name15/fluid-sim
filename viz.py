import moderngl
import moderngl_window as mglw 
from data import Field2D
import numpy as np
from timeit import timeit

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
        (y, x) = np.indices((gs[1], gs[0]))
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
        self.colors = np.zeros((gs[1], gs[0], 12), dtype='f4')
        
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
        self.arrows = np.empty((gs[1], gs[0], 4), dtype=np.float32)
        self.arrows[:, :, 0] = x + 0.5 # Start x
        self.arrows[:, :, 1] = y + 0.5 # Start y
        
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
        self.colors[:, :, :] = self.field.data['density'][:, :, np.newaxis] + 0.1
        self.color_vbo.write(self.colors.tobytes())

        # Update arrow positions buffer
        self.arrows[:, :, 2] = self.arrows[:, :, 0] + self.field.data['velocity_x'] # End x
        self.arrows[:, :, 3] = self.arrows[:, :, 1] + self.field.data['velocity_y'] # End y
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
    