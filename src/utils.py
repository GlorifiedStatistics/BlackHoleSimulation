import numpy as np
import re


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


def check_type(val, type_str: str, varname: str):
    """Checks that the given object is a good type
    
    Available types:
        - 'float'
        - 'int'
        - '[float|int]-[positive|negative|non-negative]'
    """
    type_str = type_str.lower().replace(' ', '').replace('_', '').replace('-', '').replace('integer', 'int')

    if type_str.startswith('float'):
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError("`%s` must be a %s, got: %s" % (varname, repr(type_str), repr(type(val).__name__)))
        
        if '-positive' in type_str and val <= 0:
            raise ValueError("`%s` must be positive, got: %s" % (varname, val))
        if '-non-negative' in type_str and val < 0:
            raise ValueError("`%s` must be non-negative, got: %s" % (varname, val))
        if '-negative' in type_str and val >= 0:
            raise ValueError("`%s` must be negative, got: %s" % (varname, val))
        
        return float(val)

    elif type_str.startswith('int'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError("`%s` must be a %s, got: %s" % (varname, repr(type_str), repr(type(val).__name__)))
        
        if '-positive' in type_str and val <= 0:
            raise ValueError("`%s` must be positive, got: %s" % (varname, val))
        if '-non-negative' in type_str and val < 0:
            raise ValueError("`%s` must be non-negative, got: %s" % (varname, val))
        if '-negative' in type_str and val >= 0:
            raise ValueError("`%s` must be negative, got: %s" % (varname, val))
        
        return float(val)
    
    elif type_str.startswith('point'):
        if not isinstance(val, (list, tuple)):
            raise ValueError("`%s` must be a list/tuple, got: %s" % (varname, repr(type_str), repr(type(val).__name__)))
        
        if len(val) != 3:
            raise ValueError("`%s` must be a 3-d point and thus contain 3 elements, got: %s" % (varname, val))
        
        return tuple(check_type(v, type_str='float', varname=varname+'-elem_%d' % i) for i, v in enumerate(val))
    
    else:
        raise NotImplementedError