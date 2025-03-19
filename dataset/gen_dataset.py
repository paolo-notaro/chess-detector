import bpy
import os
import math
import random
import json
import chess.pgn
import hashlib
import io
from chess import Board, SQUARES, square_file, square_rank

# Dataset
GAMES = r"lichess_processed.pgn"

# File paths
BLENDER_EXECUTABLE = r"C:/Program Files/Blender Foundation/Blender 4.3/blender.exe"
BLENDER_SCENE = os.path.abspath(r"./ChessBoard/Source/Chess Board.blend")
RENDER_PATH = os.path.abspath(r"./images/")
os.makedirs(RENDER_PATH, exist_ok=True)

METADATA_FILE = os.path.abspath(r"./metadata.json")

# Generation settings
BATCH_SIZE = 1 # Boards processed before saving state and metadata

# Load the scene
bpy.app.binary_path = BLENDER_EXECUTABLE
bpy.ops.wm.open_mainfile(filepath=BLENDER_SCENE)

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
LIGHT_INTENSITY_RANGE = (20, 100) # Watts
LIGHT_COLOR_RANGE_RGB = ((0.9, 1), (0.8, 1) ,(0.6, 1)) # R variability, G variability, B variability

AMBIENT_LIGHT_INTENSITY_RANGE = (0.2, 0.9) # Watts

# Table materials
TABLE_OBJ_NAME = "Table"
TABLE_MATERIALS = ["Rosewood"]

# Chess set materials
BOARD_OBJ_NAME = "ChessBoard"
CHESS_SET_MATERIALS = ["GreenWhite", "Wood"]

def setup_scene():

    # Set ambient light intensity (world surface)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random.uniform(*AMBIENT_LIGHT_INTENSITY_RANGE)

    # Set point light intensity, color and position
    light = bpy.data.objects[LIGHT_NAME]
    light.data.energy = random.uniform(*LIGHT_INTENSITY_RANGE)
    light.data.color = (random.uniform(*LIGHT_COLOR_RANGE_RGB[0]), random.uniform(*LIGHT_COLOR_RANGE_RGB[1]), random.uniform(*LIGHT_COLOR_RANGE_RGB[2]))
    light.location = (random.uniform(*LIGHT_POSITION_RANGE_XYZ[0]), random.uniform(*LIGHT_POSITION_RANGE_XYZ[1]), random.uniform(*LIGHT_POSITION_RANGE_XYZ[2]))

    # Select random table surface material
    table = bpy.data.objects[TABLE_OBJ_NAME]
    table.data.materials.clear()
    table.data.materials.append(bpy.data.materials[random.choice(TABLE_MATERIALS) + "_Table"])

    # Select random chess set material
    chessSet = random.choice(CHESS_SET_MATERIALS)
    
    # Apply to board
    board = bpy.data.objects[BOARD_OBJ_NAME]
    board.data.materials.clear()
    board.data.materials.append(bpy.data.materials[chessSet + "_Board"])

    # Apply to pieces
    for piece in PIECES_OBJ_NAMES.values():
        bpy.data.objects[piece].data.materials.clear()
        bpy.data.objects[piece].data.materials.append(bpy.data.materials[chessSet + "_Pieces_" + piece.split("_")[-1]])





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


def render_image(image_name):
    bpy.context.scene.render.filepath = os.path.join(RENDER_PATH, image_name)
    bpy.ops.render.render(write_still=True)

def clear_scene():
    # delete all objects in the "Temp" collection
    for obj in bpy.data.collections["Temp"].objects:
        bpy.data.objects.remove(obj)

def render_board(board, board_id):
    setup_scene()
    arrange_pieces(board)
    render_image(f"{board_id}.png")
    clear_scene()


def process_board(board, metadata):
    board_fen = board.board_fen().strip()
    board_id = hashlib.md5(board_fen.encode()).hexdigest() # Generate a unique ID for the board
    
    if not board_id in metadata["boards"]:
        metadata["boards"][board_id] = {"fen": board_fen, "lines": {}}
    
    if os.path.exists(os.path.join(RENDER_PATH, f"{board_id}.png")):
        print(f"Image for board '{board_id}' already exists. Skipping rendering.")
    else:
        render_board(board, board_id)
        print(f"Generated board '{board_id}'.")
    
    return board_id

def process_game(pgn_str, metadata):
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str[:-1]))
    except:
        return False
    board = Board()
    prev_board_id = process_board(board, metadata)
    for move in game.mainline_moves():
        board.push(move)
        new_board_id = process_board(board, metadata)

        # Save the move in the metadata
        move_str = move.uci()

        if move_str not in metadata["boards"][prev_board_id]["lines"]:
            metadata["boards"][prev_board_id]["lines"][move_str] = new_board_id

        prev_board_id = new_board_id
    
    return True

if __name__ == "__main__":
    
    if(os.path.exists(METADATA_FILE)):
        with open(METADATA_FILE, "r") as metadata_file:
            metadata = json.load(metadata_file)
    else:
        metadata = {"last_batch_end": 0, "boards" : {}}

    with open(GAMES, "r") as games_file:
        games = games_file.readlines()[metadata["last_batch_end"]:]

    while metadata["last_batch_end"] < len(games):
        start = metadata["last_batch_end"]
        end = min(metadata["last_batch_end"] + BATCH_SIZE, len(games))
        print(f"Processing game #{start}" if start == end - 1 else f"Processing games {start} to {end - 1}...")

        for i in range(start, end):
            if process_game(games[i], metadata):
                print(f"Processed game {i}.")
            else:
                print(f"PGN of game {i} could not be parsed. Skipping.")

        metadata["last_batch_end"] += BATCH_SIZE
        print("Saving metadata...")
        with open(METADATA_FILE, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        