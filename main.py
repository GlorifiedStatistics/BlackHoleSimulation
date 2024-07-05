import pygame
import src.arrays as ar
import torch
from timeit import default_timer
from src.utils import make_RGBA
from src.world import World
from src.camera import Camera, ConwaysGOLCamera

# Set which array package we will be using
ar.set_array_package('numpy')

pygame.init()

# Set up the drawing window
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 1200
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

# Set up our world/camera
ws = 100
init_state = (ar.random((ws, ws)) < 0.4).astype(int)
cameras = [ConwaysGOLCamera(init_state)]
world = World().add_objects(*cameras)
pixels = ar.full([WINDOW_WIDTH, WINDOW_HEIGHT], make_RGBA(0, 0, 255, 255), dtype='uint32')

# Handle time between updates
last_time = default_timer()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    time_inc = default_timer() - last_time
    last_time = default_timer()
    
    # Update the universe
    ut = default_timer()
    world.update(time_inc)
    print("Update time: %.4f" % (default_timer() - ut))

    # Draw all of the cameras to the current pixels
    ut = default_timer()
    for camera in cameras:
        camera.draw(pixels, world)
    print("Draw time: %.4f" % (default_timer() - ut))

    # Put the current pixels on the screen
    ut = default_timer()
    pygame.surfarray.blit_array(screen, ar.to_numpy(pixels))
    pygame.display.update()
    print("Blit time: %.4f" % (default_timer() - ut))

pygame.quit()