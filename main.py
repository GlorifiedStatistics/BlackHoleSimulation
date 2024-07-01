import pygame
import arrays as ar
from utils import make_RGBA

pygame.init()

# Set up the drawing window
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

pixels = ar.full([WINDOW_WIDTH, WINDOW_HEIGHT], make_RGBA(0, 0, 255, 255), dtype='uint32')

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    pygame.surfarray.blit_array(screen, pixels)
    pygame.display.update()

pygame.quit()