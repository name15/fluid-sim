from time import sleep
from data import Field2D

class FluidSim:
    def __init__(self, field: Field2D):
        self.field = field
    
    def step(self):
        self.field.data['velocity_x'] += 0.5
        self.field.data['velocity_y'] += 0.5
        self.field.data['density'] += 0.01
        sleep(0.1)
        