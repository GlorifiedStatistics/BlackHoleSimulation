"""Allows for switching between different array types/operations"""
from .utils import get_torch_dtype


#####################
# Set Array Package #
#####################


_ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = None, 'None'

def set_array_package(package):
    """Sets the array package to use
    
    Args:
        package (str): which package to use. Can currently support: 'numpy', 'cupy', 'torch'
    """
    global _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME
    if package in ['numpy', 'np']:
        import numpy
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = numpy, 'numpy'
    elif package in ['cupy']:
        import cupy
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = cupy, 'cupy'
    elif package in ['torch']:
        import torch
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = torch, 'torch'
    else:
        raise ValueError("Unknown array package: %s" % repr(package))
    

def get_array_package_string():
    """Returns the string name of the current array package"""
    return _ARRAY_PACKAGE_NAME


for s in ['numpy', 'cupy', 'torch']:
    try:
        set_array_package(s)
        break
    except ImportError:
        pass
else:
    raise ImportError("Could not find a valid array package to import")    


##################
# Array Creation #
##################


def empty(shape, dtype='int32'):
    """Create an uninitialized array of shape `shape`
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        dtype (Dtype): the dtype to use, defaults to int32
    """
    return _ARRAY_PACKAGE.zeros(shape, dtype=make_dtype(dtype))


def zeros(shape, dtype='int32'):
    """Create an array of shape `shape` filled with zeros
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        dtype (Dtype): the dtype to use, defaults to int32
    """
    return _ARRAY_PACKAGE.zeros(shape, dtype=make_dtype(dtype))


def full(shape, value, dtype=None):
    """Create a new array filled with the given value
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        value (Union[int, float, str]): the value to fill the array with
        dtype (Optional[Dtype]): the dtype to use
    """
    return _ARRAY_PACKAGE.full(shape, value, dtype=dtype)


def array(obj, dtype=None):
    """Create a new array from the given object
    
    Args:
        obj (ArrayLike): the object to make into an array
        dtype (Optional[Dtype]): the dtype to use
    """
    return _ARRAY_PACKAGE.array(obj, dtype=dtype)


def random(shape):
    """Random floats in range [0, 1] of given shape"""
    if _ARRAY_PACKAGE.__name__ == 'numpy':
        return _ARRAY_PACKAGE.random.rand(*shape)
    else:
        raise NotImplementedError



######################
# Inplace Operations #
######################


def fill_inplace(arr, value):
    """Fills the given array inplace with the given value"""
    return arr.fill(value)


#########################
# Vectorized Operations #



################
# Random Utils #
################


def shape(arr, dim=None):
    """Returns the shape of the given array along the given dimension
    
    Args:
        arr (Array): the array to get shape of
        dim (Optional[int]): optional dimension, or None to get full shape
    """
    return arr.shape[dim] if dim is not None else arr.shape


def count_nonzero(arr):
    """Counts the number of non-zero elements"""
    return (arr != 0).astype(int).sum()


def cast(arr, dtype):
    """Casts the array to the given dtype"""
    if get_array_package_string() in ['torch']:
        return arr.type(get_torch_dtype(dtype))
    return arr.astype(dtype)


def make_dtype(dtype):
    """Gets the dtype for the current package
    
    Args:
        dtype (str): the dtype to get
    """
    if get_array_package_string() in ['torch']:
        return get_torch_dtype(dtype)
    return _ARRAY_PACKAGE.dtype(dtype)


def to_numpy(arr):
    """Converts the array into a numpy array
    
    This may return the original array, or a view of the array data, or a copy of the array
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Could not import necessary library 'numpy' for to_numpy() conversion")
    
    if isinstance(arr, np.ndarray):
        return arr
    
    try:
        import cupy

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass

    try:
        import torch
        
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass

    return np.array(arr)