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
    def __init__(self, start_state):
        self.curr_state = start_state
        self.update_time = 0.1  # Time to update board
        self.last_update = default_timer()
        self.tested = False
    
    def update(self, world, delta):
        if default_timer() - self.last_update < self.update_time or self.tested:
            return
        self.last_update = default_timer()
        
        new_state = ar.zeros(ar.shape(self.curr_state))

        for r in range(ar.shape(self.curr_state, 0)):
            for c in range(ar.shape(self.curr_state, 1)):
                chunk = self.curr_state[max(r-1, 0):min(r+2, ar.shape(self.curr_state, 0)), max(c-1, 0):min(c+2, ar.shape(self.curr_state, 1))]
                num_alive = ar.count_nonzero(chunk)

                # If this cell is alive
                if self.curr_state[r, c] > 0:
                    
                    # If there are < 2 neighbors, or > 3 neighbors, dead, otherwise alive
                    val = 3 <= num_alive <= 4

                # If this cell is dead
                elif self.curr_state[r, c] <= 0:
                    # if there are 3 alive in chunk (all must be neighbors), make alive
                    val = num_alive == 3
                
                new_state[r, c] = val
        
        self.curr_state = new_state
    
    def draw(self, screen, world):
        line_thickness = 2
        background_color = make_RGBA(20, 20, 20, 255)
        line_color = make_RGBA(230, 230, 230, 255)
        alive_color = make_RGBA(200, 200, 10, 255)

        screen_size = min(ar.shape(screen, 0), ar.shape(screen, 1)) - 2*line_thickness

        board_start = int((ar.shape(screen, 0) - screen_size) / 2), int((ar.shape(screen, 1) - screen_size) / 2)

        # Draw the inside to be white
        ar.fill_inplace(screen, background_color)
        draw_rect(screen, board_start, screen_size+line_thickness, screen_size+line_thickness, line_color)

        cell_size = int((screen_size - ar.shape(self.curr_state, 0) * line_thickness) / ar.shape(self.curr_state, 0))

        # Draw all the squares
        for r in range(ar.shape(self.curr_state, 0)):
            for c in range(ar.shape(self.curr_state, 1)):
                loc = (line_thickness + r * (line_thickness + cell_size) + board_start[0]), \
                    (line_thickness + c * (line_thickness + cell_size) + board_start[1])
                color = alive_color if self.curr_state[c, r] > 0 else background_color  # This is reversed in lookup
                draw_rect(screen, loc, cell_size, cell_size, color)
        
        # Re-draw some black
        val = ar.shape(self.curr_state, 0) * (line_thickness + cell_size) + line_thickness
        draw_rect(screen, (board_start[0] + val, board_start[1]), screen_size+line_thickness, screen_size+line_thickness, background_color)
        draw_rect(screen, (board_start[0], val + board_start[1]), screen_size+line_thickness, screen_size+line_thickness, background_color)