import os
import sys
from contextlib import contextmanager
import chess
import chess.pgn
from chess import Board, SQUARES, square_file, square_rank
import bpy
import random 
import math

BLENDER_EXECUTABLE = r"C:/Program Files/Blender Foundation/Blender 4.3/blender.exe"
BLENDER_SCENE = os.path.abspath(r"./ChessBoard/Source/Chess Board.blend")

# Lookup tables and object names
PIECES_OBJ_NAMES = {
    "r": "Rook_Dark",
    "n": "Knight_Dark",
    "b": "Bishop_Dark",
    "q": "Queen_Dark",
    "k": "King_Dark",
    "p": "Pawn_Dark",
    "R": "Rook_Light",
    "N": "Knight_Light",
    "B": "Bishop_Light",
    "Q": "Queen_Light",
    "K": "King_Light",
    "P": "Pawn_Light"
}
LIGHT_NAME = "Point"


# Scene settings
SQUARE_SIZE = 0.128
PIECE_Z_COORD = 0.014951
A1_POS = (0.12332, 0.12332)

# Randomization settings
PIECE_PLACEMENT_VARIABILITY = 0.15 # ratio of square size

LIGHT_POSITION_RANGE_XYZ = ((-0.5, 1.7), (-0.5, 1.7), (0.7, 1.7)) # meters, X variability, Y variability, Z variability
LIGHT_INTENSITY_RANGE = (20, 70) # Watts
LIGHT_COLOR_RANGE_RGB = ((0.7, 0.9), (0.8, 0.9) ,(0.6, 0.9)) # R variability, G variability, B variability

# Table materials
TABLE_OBJ_NAME = "Table"
TABLE_MATERIALS = ["Rosewood", "Fabric"]

# Chess set materials
BOARD_OBJ_NAME = "ChessBoard"
CHESS_SET_MATERIALS = ["GreenWhite", "Wood"]


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def setup_blender():
    # Load the scene
    bpy.app.binary_path = BLENDER_EXECUTABLE
    bpy.ops.wm.open_mainfile(filepath=BLENDER_SCENE)
    bpy.data.scenes[0].render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
        

def setup_scene(chess_set = None):

    """ 
        Set up the scene with random lighting and the provided materials (or random if not provided).
        Returns the chess set material
    """

    # Set point light intensity, color and position
    light = bpy.data.objects[LIGHT_NAME]
    light.data.energy = random.uniform(*LIGHT_INTENSITY_RANGE)
    light.data.color = (random.uniform(*LIGHT_COLOR_RANGE_RGB[0]), random.uniform(*LIGHT_COLOR_RANGE_RGB[1]), random.uniform(*LIGHT_COLOR_RANGE_RGB[2]))
    light.location = (random.uniform(*LIGHT_POSITION_RANGE_XYZ[0]), random.uniform(*LIGHT_POSITION_RANGE_XYZ[1]), random.uniform(*LIGHT_POSITION_RANGE_XYZ[2]))

    # Select random table surface material
    table = bpy.data.objects[TABLE_OBJ_NAME]
    table.data.materials.clear()
    table.data.materials.append(bpy.data.materials[random.choice(TABLE_MATERIALS) + "_Table"])

    # Select random chess set material, if not specified
    if chess_set is None:
        chess_set = random.choice(CHESS_SET_MATERIALS)
    
    # Apply to board
    board = bpy.data.objects[BOARD_OBJ_NAME]
    board.data.materials.clear()
    board.data.materials.append(bpy.data.materials[chess_set + "_Board"])

    # Apply to pieces
    for piece in PIECES_OBJ_NAMES.values():
        bpy.data.objects[piece].data.materials.clear()
        bpy.data.objects[piece].data.materials.append(bpy.data.materials[chess_set + "_Pieces_" + piece.split("_")[-1]])

    return chess_set



def arrange_pieces(board : Board):
    for square in SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_obj = bpy.data.objects[PIECES_OBJ_NAMES[piece.symbol()]]

            # duplicate the object
            new_obj = piece_obj.copy()
            new_obj.data = piece_obj.data.copy()
            bpy.context.collection.objects.link(new_obj)

            # place it in the "Temp" collection
            #bpy.context.scene.collection.children["Temp"].objects.link(new_obj)

            # set the location of the object
            x = A1_POS[0] + SQUARE_SIZE * square_file(square) + random.uniform(-PIECE_PLACEMENT_VARIABILITY, PIECE_PLACEMENT_VARIABILITY) * SQUARE_SIZE 
            y = A1_POS[1] + SQUARE_SIZE * square_rank(square) + random.uniform(-PIECE_PLACEMENT_VARIABILITY, PIECE_PLACEMENT_VARIABILITY) * SQUARE_SIZE
            z = PIECE_Z_COORD

            # rotate piece randomly on the z-axis, using its origin as the pivot point
            new_obj.rotation_euler = (0, 0, random.uniform(0, 2*math.pi))

            new_obj.location = (x, y, z)


def render_image(path, image_name, suppress_output = True):
    if suppress_output:
        with stdout_redirected():
            bpy.context.scene.render.filepath = os.path.join(path, image_name)
            bpy.ops.render.render(write_still=True)
    else:
        bpy.context.scene.render.filepath = os.path.join(path, image_name)
        bpy.ops.render.render(write_still=True)

def clear_scene():
    # delete all objects in the "Temp" collection
    for obj in bpy.data.collections["Temp"].objects:
        bpy.data.objects.remove(obj)

def render_board(board, path, board_id):
    arrange_pieces(board)
    render_image(path, f"{board_id}.png")
    clear_scene()

def get_board_id(board : Board | str):
    board_fen = board.board_fen() if isinstance(board, Board) else board
    return board_fen.strip().replace("/", "_")

def process_board(board, path):
    board_id = get_board_id(board)
    
    filepath = os.path.join(path, f"{board_id}.png")

    if os.path.exists(filepath):
        print(f"Image for board '{board_id}' already exists. Skipping rendering.")
    else:
        render_board(board, path, board_id)
        print(f"Generated board '{board_id}'.")
    
    return board_id

def generate_base_boards(path):
    clear_scene()
    # For each material whose empty board isn't generated yet, generate it
    for material in CHESS_SET_MATERIALS:
        if not os.path.exists(os.path.join(path, f"empty_board_{material}.png")):
            print(f"Generating empty board for material '{material}'...")
            setup_scene(material)
            render_image(path, f"empty_board_{material}.png")

    return os.path.join(path, "empty_board_*.png")