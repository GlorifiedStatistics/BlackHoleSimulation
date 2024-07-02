from .world_object import WorldObject
from typing import Optional


class Sphere(WorldObject):
    """A solid sphere
    
    Parameters
    ----------
    size: `Optional[float]`
        Optional size of this sphere. Defaults to 1.0
    """
    size: float = 1.0

    def __init__(self, size: Optional[float] = None):
        self.size = size if size is not None else self.size