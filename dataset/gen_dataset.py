import os
import download
import analysis
import csv
import rendering
from chess import Board
import postprocessing
import random

# Dataset
GAMES = r"lichess_processed.pgn"
MOVES = r"entries.csv"

# File paths
RENDER_PATH = os.path.abspath(r"./images/")
PREPROCESS_PATH = os.path.abspath(r"./preprocessed/")
LAST_PROCESSED_INDEX_FILE = os.path.abspath(r"./last_index.txt")

METADATA_FILE = os.path.abspath(r"./metadata.json")
ERRORS_FILE = os.path.abspath(r"./errors.txt")

# Generation settings
BATCH_SIZE = 1 # Entries processed before saving state

def get_moves():
    if not os.path.exists(MOVES):
        download.download_from_lichess(name = "lichess_db_standard_rated_2016-03", output_path = "lichess_truncated.pgn", keep_n = 10000)
        sorted_moves = analysis.analyze_games("lichess_truncated.pgn")
        print(f"Total of {sorted_moves['total']} moves found. What % of moves should we keep (out of 100)?")
        
        n_moves = int(input())

        print(f"Do you want to set a minimum count for each move type? (number or empty)")

        typed = input()

        min_count = int(typed) if typed != "" else 0

        moves = list(analysis.select(sorted_moves, n_moves / 100, n_moves / 100, n_moves / 100, n_moves / 100, min_count = min_count, print_info = True))

        print("Shuffling moves...")
        random.shuffle(moves)

        print("Saving moves...")
        with open(MOVES, "w") as moves_file:
            moves_file.write("before_fen,move_uci,after_fen\n")
            for before_id, move, after_id in moves:
                moves_file.write(f"{before_id},{move},{after_id}\n")

    with open(MOVES, "r") as moves_file:
        return list(csv.reader(moves_file))[1:]

                

if __name__ == "__main__":
    
    os.makedirs(RENDER_PATH, exist_ok=True)
    os.makedirs(PREPROCESS_PATH, exist_ok=True)

    moves = get_moves()

    print(f"Total of {len(moves)} moves found.")

    print("Setting up Blender...")

    rendering.setup_blender()

    # Check if base board generation is needed
    base_board_path_match = rendering.generate_base_boards(RENDER_PATH)

    chessboard_corners = postprocessing.calibrate_camera(base_board_path_match)

    if os.path.exists(LAST_PROCESSED_INDEX_FILE):
        with open(LAST_PROCESSED_INDEX_FILE, "r") as f:
            last_processed_index = int(f.read())
    else:
        last_processed_index = 0

    i = last_processed_index

    while i < len(moves):
        print(f"Processing move {i}/{len(moves)}...")

        before_fen, move_uci, after_fen = moves[i]

        try:
            before_board = Board(before_fen)
            after_board = Board(after_fen)
        except:
            with open(ERRORS_FILE, "a") as f:
                f.write(f"Error parsing FENs: {before_fen}, {after_fen}\n")
            continue

        try:
            rendering.setup_scene()

            before_board_id = rendering.process_board(before_board, RENDER_PATH)
            after_board_id = rendering.process_board(after_board, RENDER_PATH)

            postprocessing.process_image(os.path.join(RENDER_PATH, f"{before_board_id}.png"), os.path.join(PREPROCESS_PATH, f"{before_board_id}.png"), chessboard_corners)
            postprocessing.process_image(os.path.join(RENDER_PATH, f"{after_board_id}.png"), os.path.join(PREPROCESS_PATH, f"{after_board_id}.png"), chessboard_corners)
        except:
            with open(ERRORS_FILE, "a") as f:
                f.write(f"Error processing images: {before_fen}, {after_fen}\n")
            continue
        
        if i % BATCH_SIZE == 0:

            with open(LAST_PROCESSED_INDEX_FILE, "w") as f:
                f.write(str(i))

        i += 1