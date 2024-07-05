
def make_RGBA(r, g, b, a):
    """Convert (r, g, b, a) into unsigned int"""
    return (a << 24) + (r << 16) + (g << 8) + b


def draw_rect(screen, pos, width, height, color):
    """Draws a rectangle to the given screen"""
    screen[pos[0]:pos[0]+width, pos[1]:pos[1]+height] = color


def get_torch_dtype(dtype):
    """Returns a pytorch dtype for the given dtype. I'm sure there's a function to do this somewhere, but I couldn't find it..."""
    import torch

    if isinstance(dtype, type):
        if dtype is float:
            return torch.float32
        elif dtype is int:
            return torch.int32
        else:
            raise ValueError("Unknown dtype type: %s" % repr(dtype.__name__))
    
    elif isinstance(dtype, str):
        d = dtype.lower().strip().replace('integer', 'int').replace("_", "").replace(" ", '').replace('unsigned', 'u').replace("boolean", 'bool')

        if d in (['float', 'int', 'double', 'half', 'chalf', 'cfloat', 'cdouble', 'bool', 'short', 'int', 'long'] +
                 ['int%d' % v for v in [8, 16, 32, 64]] + 
                 ['uint%d' % v for v in [8, 16, 32, 64]] + 
                 ['float%d' % v for v in [16, 32, 64]] +
                 ['complex%d' % v for v in [32, 64, 128]]):
            return getattr(torch, d)
            
        else:
            raise ValueError("Unknown dtype string: %s" % repr(dtype))

    else:
        raise TypeError("Cannot parse dtype of type: %s" % repr(type(dtype).__name__))
