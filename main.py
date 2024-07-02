import pygame
import src.arrays as ar
import torch
from src.utils import make_RGBA
from src.world import World
from src.camera import Camera

# Set which array package we will be using
ar.set_array_package('cupy')

pygame.init()

# Set up the drawing window
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

# Set up our world/camera
cameras = [Camera(position=(0, 0, 0))]
world = World().add_objects(*cameras)
pixels = ar.full([WINDOW_WIDTH, WINDOW_HEIGHT], make_RGBA(0, 0, 255, 255), dtype='uint32')

# Handle time increments
time_inc = 1.0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update the universe
    world.update(time_inc)

    # Draw all of the cameras to the current pixels
    for camera in cameras:
        camera.draw(pixels, world)

    # Put the current pixels on the screen
    pygame.surfarray.blit_array(screen, ar.to_numpy(pixels))
    pygame.display.update()

pygame.quit()