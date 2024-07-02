"""Camera class to be placed in a world"""
from . import arrays as ar
from .utils import make_RGBA
from .objects import WorldObject
from typing_extensions import Self, Optional


class Camera(WorldObject):
    """Camera class that can be placed in a world to look at things
    

    Parameters
    ----------
    position: `Optional[tuple[float, float, float]]`
        Optional initial position of the camera in 3D space. If not passed, defaults to (0, 0, 0)
    """
    def __init__(self, position: Optional[tuple[float, float, float]] = None):
        super().__init__()
        self.set_position(*position)

    def draw(self, screen, world):
        """Draws what the camera currently sees to the given screen"""
        ar.fill_inplace(screen, make_RGBA(255, 0, 0, 255))
