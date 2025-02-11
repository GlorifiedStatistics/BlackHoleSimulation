"""Holds objects/cameras and simulates reality"""
from .objects import WorldObject
from typing_extensions import Self

class World:
    """Holds objects/cameras and simulates reality
    
    Parameters
    ----------
    """

    objects: list[WorldObject] = []
    """The objects in the world"""

    time: float = 0.0
    """The current time in the world"""
    
    def __init__(self):
        pass

    def add_object(self, wo: WorldObject) -> Self:
        """Adds the given object to the world"""
        if not isinstance(wo, WorldObject):
            raise TypeError("Can only add objects of type 'WorldObject', not %s" % repr(type(wo).__name__))
        self.objects.append(wo)
        return self
    
    def add_objects(self, *objs: WorldObject) -> Self:
        """Adds all of the given objects to the world"""
        for obj in objs:
            self.add_object(obj)
        return self
    
    def update(self, delta: float):
        """Updates the universe with the given amount of time passing"""
        for wo in self.objects:
            wo.update(self, delta)
        self.time += delta
        