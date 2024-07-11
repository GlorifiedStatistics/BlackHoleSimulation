"""Camera class to be placed in a world"""
from ..utils import check_type, make_RGBA
from .. import arrays as ar
from ..world import World


# Dtype for numpy ray tracing
_NP_RT_DTYPE = 'float32'


class RayTracingCamera:
    """Camera that performs ray tracing

    This camera is immovable, always located at (0, 0, 0), with the center of the viewport being located at (0, 0, focal_length).

    If you wish to move the camera, you can instead rotate the entire world around it. Very self-centered, this camera is
    
    Parameters
    ----------
    focal_length: `float`
        The focal length of the camera.
    viewport_width: `float`
        The width of the viewport in world size. The height will fit the aspect ratio of the screen during draw() calls
    array_package: `str`
        The array package to use
    """
    def __init__(self, focal_length: float = 1.0, viewport_width: float = 2.0, array_package: str = 'numpy'):
        
        self.focal_length = check_type(focal_length, 'float-positive', varname='focal_length')
        self.viewport_width = check_type(viewport_width, 'float-positive', varname='viewport_width')

        with ar.array_package_context(array_package):
            self.array_package = ar.get_array_package_string()

    def draw(self, screen, world):
        """Draws what the camera currently sees to the given screen"""
        if self.array_package in ['numpy']:
            self._draw_numpy(screen, world)
        else:
            raise NotImplementedError

    @ar.array_package_decorator('numpy')
    def _draw_numpy(self, screen, world: World):
        """Super slow numpy/python version of ray tracing"""
        # Find the viewport height, same aspect ratio as screen, using our self.viewport_width
        viewport_height = ar.shape(screen, 0) * self.viewport_width / ar.shape(screen, 1)

        # The side length of a virtual 'pixel' on the viewport in space
        viewport_pix_len = self.viewport_width / ar.shape(screen, 1)

        # The starting point of our ray, always (0, 0, 0)
        ray_start = ar.array([0, 0, 0], dtype=_NP_RT_DTYPE)

        # Maximum distance before reaching edge of the universe (used for selecting color right now)
        max_distance = 10.0

        # Go through each pixel in the screen
        for row_i in range(ar.shape(screen, 0)):
            for col_i in range(ar.shape(screen, 1)):

                # Compute our ray endpoint. Start is (0, 0, 0), end is the center of the virtual 'pixel' on 
                #   the viewport in space
                # We add 0.5 to the row/column inds to shift into center of pixel
                # We also have to flip the rows around, otherwise the camera will be upside down
                ray_end = ar.array([-self.viewport_width / 2 + (col_i + 0.5) * viewport_pix_len, 
                                    -viewport_height / 2 + ((ar.shape(screen, 0) - row_i) + 0.5) * viewport_pix_len,
                                    self.focal_length], dtype=_NP_RT_DTYPE)
                
                # Go through each object in the world and find its collision with our ray
                collisions = [v for v in [obj.distance(ray_start, ray_end) for obj in world.objects] if v >= 0]
                cv = (255 - int(255 * min(collisions) / max_distance)) if len(collisions) > 0 else None

                # If we have collided with anything, set its color in black/white based on distance
                # Otherwise, set color to black
                screen[row_i, col_i] = make_RGBA(cv, cv, cv, 255) if cv is not None else make_RGBA(0, 0, 0, 255)
