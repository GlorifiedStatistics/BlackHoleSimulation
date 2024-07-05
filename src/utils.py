
def make_RGBA(r, g, b, a):
    """Convert (r, g, b, a) into unsigned int"""
    return (a << 24) + (r << 16) + (g << 8) + b


def draw_rect(screen, pos, width, height, color):
    """Draws a rectangle to the given screen"""
    screen[pos[0]:pos[0]+width, pos[1]:pos[1]+height] = color
