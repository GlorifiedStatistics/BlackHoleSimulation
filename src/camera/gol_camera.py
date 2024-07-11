from timeit import default_timer
from .. import arrays as ar
from ..utils import make_RGBA


class ConwaysGOLCamera:
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
            self.cell_updates = None
            self.update_time = 0.01  # Time to update board
            self.last_update = default_timer()
            self.screen_drawn = False
    
    def update(self):
        with ar.array_package_context(self.array_package):
            self.last_update = default_timer()
            
            # Count values in neighborhood and determine new state
            int_dtype = 'int16'
            dtype = 'float16' if ar.get_array_package_string() in ['torch'] else int_dtype
            nc = ar.cast(ar.convolve2d(ar.cast(self.curr_state, dtype), ar.ones((3, 3), dtype=dtype), padding=0), int_dtype)

            new_state = ar.cast(ar.logical_or(ar.logical_and(ar.logical_not(self.curr_state), nc == 3),
                                    ar.logical_and(self.curr_state, ar.logical_and(nc >= 3, nc <= 4))), int_dtype)

            # Figure out which cells need updating/drawing. Only append to update list, don't override in case this update 
            #   happens multiple times before a draw
            # We convert to numpy here because torch's tensors are suuuuuupppperr slow when getting and using individual values
            updated = ar.argwhere(self.curr_state != new_state)
            self.cell_updates = zip(ar.to_numpy(updated), ar.to_numpy(new_state[updated[:, 0], updated[:, 1]]))
            self.curr_state = new_state
    
    @ar.array_package_decorator('numpy')
    def draw(self, screen, world, force_update=False):
        # Only update if either we are forcing it, or enough time has passed and we have drawn the previous updates
        if force_update or (default_timer() - self.last_update > self.update_time):
            self.update()
        
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
        if self.cell_updates is not None:
            for (r, c), state in self.cell_updates:
                color = alive_color if state > 0 else background_color
                screen[board_start[0]+(cell_size+line_thickness)*r+line_thickness:board_start[0]+(cell_size+line_thickness)*r+line_thickness+cell_size,
                        board_start[1]+(cell_size+line_thickness)*c+line_thickness:board_start[1]+(cell_size+line_thickness)*c+line_thickness+cell_size] = color
        
            self.cell_updates = None
