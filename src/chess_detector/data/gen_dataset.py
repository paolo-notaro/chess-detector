"""Generate a synthetic chess move dataset by rendering Blender scenes.

This module is invoked as the ``chess-detector-gen-dataset`` console script.
By default it writes its outputs under ``./dataset/`` relative to the current
working directory; override the location via the environment variable
``CHESS_DETECTOR_DATA_DIR``.
"""

from __future__ import annotations

import csv
import random

from chess import Board

from chess_detector.data import analysis, download, paths, postprocessing

_DATA_DIR = paths.data_dir()

GAMES = _DATA_DIR / "lichess_processed.pgn"
MOVES = paths.entries_file()

RENDER_PATH = paths.images_dir()
PREPROCESS_PATH = paths.preprocessed_dir()
DIFF_PATH = paths.diff_dir()
LAST_PROCESSED_INDEX_FILE = paths.last_index_file()

METADATA_FILE = _DATA_DIR / "metadata.json"
ERRORS_FILE = _DATA_DIR / "errors.txt"

BATCH_SIZE = 1
DELETE_RENDERED = False
GEN_DIFF = True
SUPPRESS_BLENDER_OUTPUT = False


def download_and_select_moves() -> set[tuple[str, str, str]]:
    """Download PGN data from Lichess, analyse it and interactively pick moves."""
    download.download_from_lichess(
        name="lichess_db_standard_rated_2016-03",
        output_path=str(_DATA_DIR / "lichess_truncated.pgn"),
        keep_n=10000,
    )
    sorted_moves, total_moves = analysis.analyze_games(str(_DATA_DIR / "lichess_truncated.pgn"))
    print(f"Total of {total_moves} moves found. What % of moves should we keep (out of 100)?")
    n_moves = int(input())

    print("Do you want to set a minimum count for each move type? (number or empty)")
    typed = input()
    min_count = int(typed) if typed != "" else 0

    moves = analysis.select(
        sorted_moves,
        n_moves / 100,
        n_moves / 100,
        n_moves / 100,
        n_moves / 100,
        min_count=min_count,
        print_info=True,
    )

    print("Shuffling moves...")
    moves_list = list(moves)
    random.shuffle(moves_list)
    return set(moves_list)


def get_moves() -> list[list[str]]:
    """Return all dataset moves, lazily creating ``entries.csv`` if missing."""
    if not MOVES.exists():
        moves = download_and_select_moves()
        print("Saving moves...")
        with MOVES.open("w") as moves_file:
            moves_file.write("before_fen,move_uci,after_fen\n")
            for before_id, move, after_id in moves:
                moves_file.write(f"{before_id},{move},{after_id}\n")
    else:
        print("An entries.csv file already exists. Do you want to append to it? (y/n)")
        if input() == "y":
            moves = download_and_select_moves()
            with MOVES.open("a") as moves_file:
                for before_id, move, after_id in moves:
                    moves_file.write(f"{before_id},{move},{after_id}\n")

    with MOVES.open("r") as moves_file:
        return list(csv.reader(moves_file))[1:]


def main() -> None:
    """Console-script entry point for dataset generation."""
    from chess_detector.data import rendering

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_PATH.mkdir(parents=True, exist_ok=True)
    PREPROCESS_PATH.mkdir(parents=True, exist_ok=True)
    if GEN_DIFF:
        DIFF_PATH.mkdir(parents=True, exist_ok=True)

    moves = get_moves()
    print(f"Total of {len(moves)} moves found.")

    print("Setting up Blender...")
    rendering.setup_blender()

    base_board_path_match = rendering.generate_base_boards(str(RENDER_PATH))
    chessboard_corners = postprocessing.calibrate_camera_from_path_match(base_board_path_match)

    if LAST_PROCESSED_INDEX_FILE.exists():
        last_processed_index = int(LAST_PROCESSED_INDEX_FILE.read_text())
    else:
        last_processed_index = 0

    i = last_processed_index
    while i < len(moves):
        print(f"Processing move {i}/{len(moves)}...")
        before_fen, _move_uci, after_fen = moves[i]

        try:
            before_board = Board(before_fen)
            after_board = Board(after_fen)
        except ValueError:
            with ERRORS_FILE.open("a") as f:
                f.write(f"Error parsing FENs: {before_fen}, {after_fen}\n")
            continue

        try:
            rendering.setup_scene()

            before_board_id = rendering.get_board_id(before_board)
            after_board_id = rendering.get_board_id(after_board)

            piece_placement_variability = rendering.gen_piece_placement_variability()

            before_target = PREPROCESS_PATH / f"{before_board_id}.png"
            if not before_target.exists():
                before_board_id = rendering.process_board(
                    before_board,
                    str(RENDER_PATH),
                    piece_placement_variability,
                    suppress_output=SUPPRESS_BLENDER_OUTPUT,
                )
                before_img = postprocessing.process_image(
                    str(RENDER_PATH / f"{before_board_id}.png"), chessboard_corners
                )
                postprocessing.save_image(before_img, str(before_target))
                if DELETE_RENDERED:
                    (RENDER_PATH / f"{before_board_id}.png").unlink(missing_ok=True)
            else:
                print(f"Processed image already exists: {before_board_id}.png")
                before_img = postprocessing.process_image(
                    str(RENDER_PATH / f"{before_board_id}.png"), chessboard_corners
                )

            after_target = PREPROCESS_PATH / f"{after_board_id}.png"
            if not after_target.exists():
                after_board_id = rendering.process_board(
                    after_board,
                    str(RENDER_PATH),
                    piece_placement_variability,
                    suppress_output=SUPPRESS_BLENDER_OUTPUT,
                )
                after_img = postprocessing.process_image(
                    str(RENDER_PATH / f"{after_board_id}.png"), chessboard_corners
                )
                postprocessing.save_image(after_img, str(after_target))
                if DELETE_RENDERED:
                    (RENDER_PATH / f"{after_board_id}.png").unlink(missing_ok=True)
            else:
                print(f"Processed image already exists: {after_board_id}.png")
                after_img = postprocessing.process_image(
                    str(RENDER_PATH / f"{after_board_id}.png"), chessboard_corners
                )

            if GEN_DIFF:
                diff_img = postprocessing.gen_diff(before_img, after_img)
                postprocessing.save_image(diff_img, str(DIFF_PATH / f"{i}.png"))
                print(f"Diff image saved: {i}_diff.png")

        except Exception as e:
            with ERRORS_FILE.open("a") as f:
                f.write(f"Error processing images: {before_fen}, {after_fen}: {e.__class__}\n")
        finally:
            if i % BATCH_SIZE == 0:
                LAST_PROCESSED_INDEX_FILE.write_text(str(i))
            i += 1


if __name__ == "__main__":
    main()
