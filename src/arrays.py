"""Allows for switching between different array types/operations"""
import cupyx.scipy.signal
from .utils import get_torch_dtype
from contextlib import contextmanager


#####################
# Set Array Package #
#####################


_ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = None, 'None'
_CUDA_DEVICE = None

def set_array_package(package):
    """Sets the array package to use
    
    Args:
        package (str): which package to use. Can currently support: 'numpy', 'cupy', 'torch', 'torch-cpu', 'torch-gpu'
    
    Returns:
        Union[None, str]: None if this is the first call to set_array_package, otherwise string
            name of the old package this was before overwriting
    """
    global _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME, _CUDA_DEVICE

    ret_val = None if _ARRAY_PACKAGE is None else _ARRAY_PACKAGE_NAME

    package = package.lower().replace('_', '-')

    if package in ['numpy', 'np']:
        import numpy
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = numpy, 'numpy'
    elif package in ['cupy']:
        import cupy
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = cupy, 'cupy'
    elif package in ['torch', 'torch-cpu', 'torch-gpu']:
        import torch
        _ARRAY_PACKAGE, _ARRAY_PACKAGE_NAME = torch, 'torch'
        _CUDA_DEVICE = 'cuda:0' if package == 'torch-gpu' or (package != 'torch-cpu' and torch.cuda.is_available()) else 'cpu'
    else:
        raise ValueError("Unknown array package: %s" % repr(package))
    
    return ret_val


@contextmanager
def array_package_context(package):
    """Context manager to set array package temporarily"""
    try:
        old_package = set_array_package(package)
        yield
    finally:
        set_array_package(old_package)


def array_package_decorator(package):
    """Decorate a function to set the array package while within that function"""
    def wrap(func):
        def new_func(*args, **kwargs):
            with array_package_context(package):
                func(*args, **kwargs)
        return new_func
    return wrap


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
    return set_gpu_device(_ARRAY_PACKAGE.empty(tuple(shape), dtype=make_dtype(dtype)))


def zeros(shape, dtype='int32'):
    """Create an array of shape `shape` filled with zeros
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        dtype (Dtype): the dtype to use, defaults to int32
    """
    return set_gpu_device(_ARRAY_PACKAGE.zeros(tuple(shape), dtype=make_dtype(dtype)))


def ones(shape, dtype='int32'):
    """Create an array of shape `shape` filled with ones
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        dtype (Dtype): the dtype to use, defaults to int32
    """
    return set_gpu_device(_ARRAY_PACKAGE.ones(tuple(shape), dtype=make_dtype(dtype)))


def full(shape, value, dtype=None):
    """Create a new array filled with the given value
    
    Args:
        shape (Iterable[int]): dimensions of empty array
        value (Union[int, float, str]): the value to fill the array with
        dtype (Optional[Dtype]): the dtype to use
    """
    return set_gpu_device(_ARRAY_PACKAGE.full(tuple(shape), value, dtype=make_dtype(dtype)))


def array(obj, dtype=None):
    """Create a new array from the given object
    
    Args:
        obj (ArrayLike): the object to make into an array
        dtype (Optional[Dtype]): the dtype to use
    """
    if _ARRAY_PACKAGE_NAME in ['torch']:
        return set_gpu_device(_ARRAY_PACKAGE.tensor(obj, dtype=make_dtype(dtype)))
    return _ARRAY_PACKAGE.array(obj, dtype=make_dtype(dtype))


def random(shape):
    """Random floats in range [0, 1] of given shape"""
    shape = tuple(shape)

    if _ARRAY_PACKAGE_NAME in ['numpy']:
        return _ARRAY_PACKAGE.random.rand(*shape)
    elif _ARRAY_PACKAGE_NAME in ['torch']:
        return set_gpu_device(_ARRAY_PACKAGE.rand(shape))
    else:
        raise NotImplementedError


def padded(arr, n_rows, n_cols, pad_val):
    """Pads the given array with the number of rows/cols filled with pad_val"""
    if ndim(arr) != 2:
        raise ValueError("Can only pad 2-d arrays")
    
    ret = full((shape(arr, 0) + 2*n_rows, shape(arr, 1) + 2*n_cols), pad_val)
    ret[n_rows:-n_rows, n_cols:-n_cols] = arr
    return set_gpu_device(ret)


######################
# Inplace Operations #
######################


def fill_inplace(arr, value):
    """Fills the given array inplace with the given value"""
    return set_gpu_device(arr.fill(value))


#########################
# Vectorized Operations #
#########################


def argwhere(arr):
    """Returns the places where arr is True"""
    return set_gpu_device(_ARRAY_PACKAGE.argwhere(arr))


def dot(arr1, arr2):
    """Returns the dot product of the two vectors"""
    if ndim(arr1) != 1 or ndim(arr2) != 1:
        raise ValueError("Can only do dot product on 1-d vectors!")
    return _ARRAY_PACKAGE.dot(arr1, arr2)


def convolve2d(arr, kernel, padding=None):
    """Performs a 2d convolution of kernel on arr
    
    Args:
        arr (Array): the array
        kernel (Array): the kernel to use. Should have odd side lengths
        passing (Union[str, int, None]): The padding to use. Can be:

            - None: use no padding, new shape will be smaller
            - int: value to use for padding
    """
    if ndim(arr) != 2 or ndim(kernel) != 2:
        raise ValueError("Can only perform convolve2d on 2-d arrays. Shapes: %s and %s" % (shape(arr), shape(kernel)))
    if shape(kernel, 0) % 2 != 1 or shape(kernel, 1) % 2 != 1:
        raise ValueError("Can only do convolution with odd-lengthed kernel. Kernel shape: %s" % (shape(kernel),))
    if padding is not None and not isinstance(padding, (int, float)):
        raise TypeError("Unknown padding type: %s" % repr(type(padding).__name__))
    
    
    # For numpy, we have to implement it ourselves
    if _ARRAY_PACKAGE_NAME in ['numpy']:
        
        # Deal with padding
        if isinstance(padding, (int, float)):
            arr = padded(arr, (shape(kernel, 0) // 2), (shape(kernel, 1) // 2), padding)
        elif padding is not None:
            raise NotImplementedError

        # Taken from: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
        s = shape(kernel) + tuple(_ARRAY_PACKAGE.subtract(shape(arr), shape(kernel)) + 1)
        strd = _ARRAY_PACKAGE.lib.stride_tricks.as_strided
        subM = strd(arr, shape=s, strides=arr.strides * 2)
        return _ARRAY_PACKAGE.einsum('ij,ijkl->kl', kernel, subM)
    
    elif _ARRAY_PACKAGE_NAME in ['torch']:
        import torch

        # Deal with padding
        if isinstance(padding, (int, float)) and padding != 0:
            arr = padded(arr, (shape(kernel, 0) // 2), (shape(kernel, 1) // 2), padding)
        elif padding not in [0, None]:
            raise NotImplementedError
        
        padding = 'same' if padding == 0 else 'valid'
        return torch.nn.functional.conv2d(set_gpu_device(arr.unsqueeze(0).unsqueeze(0)), 
            set_gpu_device(kernel.unsqueeze(0).unsqueeze(0)), padding=padding)[0][0]
    
    elif _ARRAY_PACKAGE_NAME in ['cupy']:
        import cupyx

        # Deal with padding
        fill_value = padding
        boundary = 'fill'
        if isinstance(padding, (int, float)):
            mode = 'same'
        elif padding is None:
            mode = 'valid'
        else:
            raise NotImplementedError

        return cupyx.scipy.signal.convolve2d(arr, kernel, mode=mode, boundary=boundary, fillvalue=fill_value)
    else:
        raise NotImplementedError
    

######################
# Logical Operations #
######################


def logical_not(arr):
    """Returns ~arr"""
    return set_gpu_device(_ARRAY_PACKAGE.logical_not(arr))


def logical_and(arr1, arr2):
    """Returns arr1 & arr2"""
    return set_gpu_device(_ARRAY_PACKAGE.logical_and(arr1, arr2))


def logical_or(arr1, arr2):
    """Returns arr1 | arr2"""
    return set_gpu_device(_ARRAY_PACKAGE.logical_or(arr1, arr2))


def logical_xor(arr1, arr2):
    """Returns arr1 ^ arr2"""
    return set_gpu_device(_ARRAY_PACKAGE.logical_xor(arr1, arr2))


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
        return set_gpu_device(arr.type(get_torch_dtype(dtype)))
    return arr.astype(dtype)


def make_dtype(dtype):
    """Gets the dtype for the current package
    
    Args:
        dtype (str): the dtype to get
    """
    if get_array_package_string() in ['torch']:
        return get_torch_dtype(dtype)
    return _ARRAY_PACKAGE.dtype(dtype)


def ndim(arr):
    "Returns the number of dimensions in the given array"
    return arr.ndim


def dtype(arr):
    """Returns the dtype of arr"""
    return arr.dtype


def set_gpu_device(arr, device=None):
    """Sends the given array to the default gpu device. Does nothing if using a non-gpu array package"""
    device = _CUDA_DEVICE if device is None else device
    
    if get_array_package_string() in ['torch']:
        return arr.to(device)
    else:
        return arr


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