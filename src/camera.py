"""Camera class to be placed in a world"""
from timeit import default_timer
from . import arrays as ar
from .utils import make_RGBA, draw_rect
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
        pass


class ConwaysGOLCamera(Camera):
    """Plays conway's game of life
    
    Parameters
    ----------
    start_state: `array`
        2d Array of board start state. Should have 1's in living cells, 0's in dead
    """
    def __init__(self, start_state, array_package='numpy'):
        self.array_package = array_package
        with ar.array_package_context(self.array_package):
            self.curr_state = ar.array(start_state, dtype='int32')
            self.cell_updates = []
            self.update_time = 0.01  # Time to update board
            self.last_update = default_timer()
            self.screen_drawn = False
    
    def update(self, world, delta):
        with ar.array_package_context(self.array_package):
            if default_timer() - self.last_update < self.update_time:
                return
            self.last_update = default_timer()
            
            # Count values in neighborhood and determine new state
            nc = ar.convolve2d(self.curr_state, ar.ones((3, 3), dtype='int32'), padding=0)
            new_state = ar.cast(ar.logical_or(ar.logical_and(ar.logical_not(self.curr_state), nc == 3),
                                    ar.logical_and(self.curr_state, ar.logical_and(nc >= 3, nc <= 4))), 'int32')

            # Figure out which cells need updating/drawing. Only append to update list, don't override in case this update 
            #   happens multiple times before a draw
            # We convert to numpy here because torch's tensors are suuuuuupppperr slow when getting and using individual values
            update_state = ar.to_numpy(new_state)
            self.cell_updates += [(r, c, update_state[r, c]) for r, c in ar.to_numpy(ar.argwhere(self.curr_state != new_state))]
            self.curr_state = new_state
    
    @ar.array_package_decorator('numpy')
    def draw(self, screen, world):
        print(ar.get_array_package_string(), type(screen), len(self.cell_updates), self.cell_updates[0] if len(self.cell_updates) > 0 else None)
        line_thickness = 0
        background_color = make_RGBA(20, 20, 20, 255)
        line_color = make_RGBA(230, 230, 230, 255)
        alive_color = make_RGBA(200, 200, 10, 255)

        # Full screen is square in middle, with some empty padding around edges (based on line thickness)
        screen_size = min(ar.shape(screen, 0), ar.shape(screen, 1)) - 2 * line_thickness

        # Size of each cell on the screen, rounded down
        cell_size = int((screen_size - max(ar.shape(self.curr_state)) * line_thickness) / max(ar.shape(self.curr_state)))

        # Compute the padding along the top/bottom
        board_size = (ar.shape(self.curr_state, 0) * (line_thickness + cell_size) + line_thickness,
                    ar.shape(self.curr_state, 1) * (line_thickness + cell_size) + line_thickness)
        board_start = ((ar.shape(screen, 0) - board_size[0]) // 2, (ar.shape(screen, 1) - board_size[1]) // 2)

        # Draw the background and cell boundaries if they haven't been drawn yet
        if not self.screen_drawn:
            ar.fill_inplace(screen, background_color)
            for r in range(ar.shape(self.curr_state, 0) + 1):
                screen[board_start[0]+(cell_size+line_thickness)*r:board_start[0]+(cell_size+line_thickness)*r+line_thickness, board_start[1]:board_start[1] + board_size[1]] = line_color
            for c in range(ar.shape(self.curr_state, 1) + 1):
                screen[board_start[0]:board_start[0] + board_size[0], board_start[1]+(cell_size+line_thickness)*c:board_start[1]+(cell_size+line_thickness)*c+line_thickness] = line_color
            self.screen_drawn = True
        
        # Draw all of the cells
        for r, c, state in self.cell_updates:
            color = alive_color if state > 0 else background_color
            screen[board_start[0]+(cell_size+line_thickness)*r+line_thickness:board_start[0]+(cell_size+line_thickness)*r+line_thickness+cell_size,
                    board_start[1]+(cell_size+line_thickness)*c+line_thickness:board_start[1]+(cell_size+line_thickness)*c+line_thickness+cell_size] = color
        
        self.cell_updates = []
