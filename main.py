from field2d import Cell, Field2D
import numpy as np
import vispy
from vispy.scene.visuals import Mesh, Arrow
import time
from threading import Thread

# Set up initial conditions
grid_size = (100, 100)

def generator(x, y):
    center = (0.25 * grid_size[0], 0.5 * grid_size[1])
    if (x - center[0])**2 + (y - center[1])**2 < 10:
        return np.array((1000, 0, 1), dtype=Cell)
    else:
        return np.array((0, 0, 0), dtype=Cell)

field = Field2D(grid_size, generator)

# Set up GUI
canvas = vispy.scene.SceneCanvas(title="Fluid demo", size=(800, 600), show=True, keys='interactive')
view = canvas.central_widget.add_view()
view.camera = vispy.scene.cameras.PanZoomCamera(aspect=1, rect=(0, 0, grid_size[0], grid_size[1]))

# Generate a colored mesh
vertices = np.indices((grid_size[0] + 1, grid_size[1] + 1)).transpose((1, 2, 0)).reshape(-1, 2)

def face_at(x, y):
    n = x * (grid_size[1] + 1) + y
    return [n, n + 1, n + grid_size[1] + 2], [n, n + grid_size[1] + 2, n + grid_size[1] + 1]

faces = np.array([face for x in range(grid_size[0]) for y in range(grid_size[1]) for face in face_at(x, y)])

def get_colors():
    densities = field.data['density']
    def color_at(x, y):
        c = min(1, max(0, densities[x, y] + 0.1))
        return [(c, c, c), (c, c, c)]

    return np.array([color for x in range(grid_size[0]) for y in range(grid_size[1]) for color in color_at(x, y)])

mesh = Mesh(vertices=vertices, faces=faces, face_colors=get_colors(), parent=view.scene)
mesh.freeze()


# Visualize the vectors of each cell
velocities_x = field.data['velocity_x']
velocities_y = field.data['velocity_y']
def velocity_at(x, y):
    return velocities_x[x, y], velocities_y[x, y]

def get_arrows():
    arrows = []
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            start = np.array([x + 0.5, y + 0.5])
            end = start + velocity_at(x, y)
            arrows.append([start, end])
    
    heads = np.concatenate(arrows, axis=0).reshape(-1, 4)
    
    return np.array(arrows).reshape(-1, 2), heads

arrows_pos, arrows_head = get_arrows()
arrow = Arrow(pos=arrows_pos, arrows=arrows_head, color='blue', width=1, method="gl", connect="segments", arrow_type="angle_30", parent=view.scene)

arrow.set_gl_state(depth_test=False)

def fixed_update():
    field.data['velocity_x'] += 0.5
    field.data['velocity_y'] += 0.5
    field.data['density'] += 0.01

    arrows_pos, arrows_head = get_arrows()
    arrow.set_data(pos=arrows_pos, arrows=arrows_head)
    mesh.set_data(vertices=vertices, faces=faces, face_colors=get_colors())

def update_graphics():
    print("Starting graphics thread")
    interval = 1.0 / 24.0
    next_time = time.monotonic()

    while True:
        now = time.monotonic()
        if now >= next_time:
            fixed_update()
            next_time += interval

        time.sleep(max(0, next_time - time.monotonic()))

Thread(target=update_graphics, daemon=True).start()

vispy.app.run()


"""
def draw_grid(field):
    s = screen.get_size()
    d = Vector2(s.x / field.size.x, s.y / field.size.y)
    
    e = min(d.x, d.y) # length of grid cell
    
    o = Vector2()
    o.x = (s.x - e * field.size.x) / 2 # horizontal offset
    o.y = (s.y - e * field.size.y) / 2 # vertical offset
    
    for y in range(int(field.size.y)):
        for x in range(int(field.size.x)):
            coords = Vector2(x, y)
            cell = field.front_buffer[coords]
            
            col = cell.density
            if col < 0: col = 0
            if col > 1: col = 1
            
            p = Vector2(x * e + o.x, y * e + o.y) # cell position
            pygame.draw.rect(
                screen,
                (col * 255, col * 255, col * 255), # color
                (p.x, p.y, e, e) # rect
            )
    
    for y in range(int(field.size.y)):
        for x in range(int(field.size.x)):
            coords = Vector2(x, y)
            cell = field.front_buffer[coords]
            
            p = Vector2(x * e + o.x, y * e + o.y) # cell position
            v = cell.velocity
            
            a1 = p + Vector2(e) / 2
            a2 = a1 + v
            a3 = a2 - v.rotate(20) / 4
            a4 = a2 - v.rotate(-20) / 4
            
            col = erf(v.length() / 25) # TODO: tweak the 25
            
            pygame.draw.line(
                screen,
                (col * 255, 0, (1 - col) * 255),
                a1, a1 + v * 0.9,
                max(int(e / 30), 1)
            )
                
            pygame.draw.polygon(
                screen,
                (col * 255, 0, (1 - col) * 255),
                [
                    a2,
                    a3,
                    a4
                ]
            )

def screen_to_grid_pos(coords, grid_size):
    s = Vector2(screen.get_size())
    d = Vector2(s.x / grid_size.x, s.y / grid_size.y)
    
    e = min(d.x, d.y)
    
    o = Vector2()
    o.x = (s.x - e * grid_size.x) / 2
    o.y = (s.y - e * grid_size.y) / 2
    
    o += Vector2(e / 2)
    
    return (coords - o) / e

########## CREATE FIELD ##########
grid_size = Vector2(60, 40)

def generator(coords):
    if coords.distance_squared_to(Vector2(0.25 * grid_size.x, 0.5 * grid_size.y)) < 10:
        return Cell(Vector2(1000, 0), 10)
    else:
        return Cell(Vector2(0, 0), 0)

field = Field2D(grid_size, generator)

###############################

i = 0
clock = pygame.time.Clock()
text_surface = None

pr = cProfile.Profile()
pr.enable()
    
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            grid_pos = screen_to_grid_pos(mouse_pos, field.size)
            
            value = field.front_buffer.interpolate_cell(grid_pos)
            text_surface = myfont.render(f"({grid_pos.x:.2f}, {grid_pos.y:.2f}): {value}", False, (255, 255, 255))
    
    clock.tick()
    fps = clock.get_fps()
    
    screen.fill((0, 0, 0))
    
    draw_grid(field)
    
    if (i == 100):
        break

    if text_surface:
        screen.blit(text_surface, (0, 0))

    if fps > 0 and i % 50 == 0:
        field.update(timestep=0.1, iterations=15, k=0.1) # IMPORTANT
    
    pygame.display.flip()
    
    i += 1

pygame.quit()

pr.disable()
stats = pstats.Stats(pr).sort_stats('cumulative')
stats.print_stats()
"""