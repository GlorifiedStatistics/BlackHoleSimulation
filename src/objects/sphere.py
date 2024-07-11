import math
from .world_object import WorldObject
from ..utils import check_type
from .. import arrays as ar


class Sphere(WorldObject):
    """A solid sphere
    
    Parameters
    ----------
    position: `tuple[float, float, float]`
        Position of this sphere in space
    radius: `float`
        radius of this sphere
    """

    def __init__(self, position: tuple[float, float, float], radius: float = 1.0):
        self.position = check_type(position, 'point', 'position')
        self.radius = check_type(radius, 'float-positive', 'radius')
    
    def update(self, world, delta):
        return super().update(world, delta)
    
    def distance(self, ray_start, ray_end):
        """See chapter 5 in https://raytracing.github.io/books/RayTracingInOneWeekend.html for derivation
        
        Use quadratic formula with:
            - a = dot(D, D), D is ray_direction
            - b = dot(-2D, C-Q), C is sphere center, Q is ray_start
            - c = dot(C-Q, C-Q) - r^2, r is radius of sphere
        """
        ray_start, ray_end = ar.array(ray_start), ar.array(ray_end)

        # Compute D and (C-Q)
        ray_direction = ray_end - ray_start
        ray_sphere_offset = ar.array(self.position) - ray_start

        # Compute a, b, and c
        a = ar.dot(ray_direction, ray_direction)
        b = ar.dot(-2*ray_direction, ray_sphere_offset)
        c = ar.dot(ray_sphere_offset, ray_sphere_offset) - self.radius ** 2

        # Compute what's under the square root of quadratic formula. If it's positive, there's 2 solutions, 0 = 1 solution
        #   (hits tangent), negative = no solutions (misses sphere)
        under_sqrt = b**2 - 4*a*c

        # If we miss, we return 1. Otherwise we return the closest value
        if under_sqrt < 0:
            return -1
        
        v1 = (-b + math.sqrt(under_sqrt)) / (2*a)
        v2 = (-b - math.sqrt(under_sqrt)) / (2*a)
        return min(v1, v2)
