from typing_extensions import Self


class WorldObject:
    """An object in the world"""

    position = (0.0, 0.0, 0.0)
    """The location of this object in space"""

    def set_position(self, x: float, y: float, z: float) -> Self:
        """Change the position of the camera"""
        self.position = (x, y, z)
        return self
    
    def update(self, world, delta):
        """Updates this object in the world"""
        pass