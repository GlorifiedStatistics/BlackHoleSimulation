
def make_RGBA(r, g, b, a):
    """Convert (r, g, b, a) into unsigned int"""
    return (a << 24) + (r << 16) + (g << 8) + b
