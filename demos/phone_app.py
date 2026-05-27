"""Tkinter demo that streams board images from an IP camera and predicts moves.

This script is invoked as the ``chess-detector-demo-phone`` console script. It
expects the ``CAMERA_URL`` constant below to be set to the URL of an IP camera
(for example, the one exposed by Android apps such as "IP Webcam").
"""

import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import chess
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageTk

from chess_detector.data import paths
from chess_detector.data.dataset import ChessMoveFromDiffDataset
from chess_detector.data.postprocessing import (
    calibrate_camera_from_images,
    gen_diff,
    rectify_board,
)
from chess_detector.inference.predict import predict_move
from chess_detector.models.diff import ChessMoveModel, ConvPatchEncoder

CAMERA_URL = os.environ.get("CHESS_DETECTOR_CAMERA_URL")
CAPTURE_CALIB_IMGS = 5

chessboard_corners = None
before_image = None
after_image = None
board = None
last_diff = None

CHECKPOINT = os.environ.get(
    "CHESS_DETECTOR_CHECKPOINT",
    str(paths.models_dir() / "checkpoint_mercurial-stag-264_epoch11.pth"),
)
ENCODER_CLASS = ConvPatchEncoder
PREPROCESSING_OUT_SIZE = 224
PATCH_SIZE = PREPROCESSING_OUT_SIZE // 8
FINAL_PATCH_SIZE = 32

SAVE_DIFF = True
SAVE_DIFF_PATH = str(paths.diff_real_dir())
SAVE_METADATA_PATH = str(paths.diff_real_metadata_file())

device: torch.device | None = None
model: ChessMoveModel | None = None
NEXT_ID = 0


def get_prediction(diff_image, board_fen, turn="wb"):

    # Resize and normalize the diff image
    preprocessed_diff_image = ChessMoveFromDiffDataset.preprocess_image(
        diff_image, preprocess_resize=PREPROCESSING_OUT_SIZE
    )

    # Split the image into patches
    patch_tensor = ChessMoveFromDiffDataset.patch_image(
        preprocessed_diff_image, resize_size=FINAL_PATCH_SIZE
    )

    moves = predict_move(model, patch_tensor, device, board_fen=board_fen, topk=5, turn=turn)

    # Print the top 5 predicted moves
    print("Top 5 predicted moves:")
    for i, (move, conf) in enumerate(moves):
        print(f"{i + 1}: {move} ({conf:.4f})")

    return moves


def get_image():
    response = requests.get(CAMERA_URL, timeout=5)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def capture_board_with_detection():
    try:
        img = get_image()
        if img is None:
            raise ValueError("Could not decode image")

        capture_board_with_detection.original_img = img.copy()

        # Chessboard detection
        corners = calibrate_camera_from_images([img])  # returns 4 corners as a numpy array

        capture_board_with_detection.chessboard_corners = corners

        # Draw bounding box of chessboard

        cv2.polylines(
            img,
            [corners.astype(np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        capture_board_with_detection.annotated_img = img
        show_image(img)

        confirm_btn.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Detection Error", str(e))


def save_last_move(last_diff):
    global NEXT_ID, board
    move = board.peek()

    # save metadata
    with open(SAVE_METADATA_PATH, "a") as f:
        f.write(f"{NEXT_ID},{move.uci()}\n")

    # save img
    img_path = os.path.join(SAVE_DIFF_PATH, f"{NEXT_ID}.png")
    cv2.imwrite(img_path, last_diff)

    NEXT_ID += 1


def prompt_user_choice(predicted_moves):
    """Prompt the user to pick a predicted move by index or type a UCI move.

    Returns the chosen UCI string. If the user cancels the dialog or submits
    an empty/whitespace-only response, the top-ranked predicted move is used.
    """
    move_strs = [f"{i + 1}: {move}" for i, (move, _) in enumerate(predicted_moves)]
    move_str = "\n".join(move_strs)

    raw = simpledialog.askstring(
        "Select Move",
        f"Select a move, leave empty for the first choice or type it in UCI form:\n{move_str}",
        parent=root,
    )

    selected = (raw or "").strip()
    if not selected:
        return predicted_moves[0][0]

    try:
        intval = int(selected)
    except ValueError:
        return selected

    if intval < 1 or intval > len(predicted_moves):
        return predicted_moves[0][0]
    return predicted_moves[intval - 1][0]


def capture_move():
    global last_diff, before_image, after_image, board

    if SAVE_DIFF and last_diff is not None:
        save_last_move(last_diff)

    try:
        img = get_image()
        if img is None:
            raise ValueError("Could not decode image")

        after_image = rectify_board(img, chessboard_corners, size=224)
        show_image(after_image)

        diff_image = gen_diff(before_image, after_image)

        predicted_moves = get_prediction(
            diff_image, board.board_fen(), turn="w" if board.turn == chess.WHITE else "b"
        )

        pred_uci = prompt_user_choice(predicted_moves)

        move = chess.Move.from_uci(pred_uci)

        while move not in board.legal_moves:
            move_str = simpledialog.askstring(
                "Edit Move",
                "The move is illegal. Enter the move in UCI format (e.g., e2e4):",
                parent=root,
                initialvalue=pred_uci,
            )
            if move_str:
                move = chess.Move.from_uci(move_str)

        board.push(move)
        update_board_display()
        manual_edit_move_btn.config(state=tk.NORMAL)
        label.config(
            text=f"Move captured. Press 'Edit Move' to modify or 'Capture' to capture again. {(board.turn == chess.WHITE and 'White') or 'Black'} to move."
        )
        manual_edit_move_btn.config(state=tk.NORMAL)
        capture_btn.config(text="Capture", command=capture_move)

        before_image = after_image.copy()  # Update before_image for the next capture
        after_image = None  # Reset after_image
        last_diff = diff_image.copy()  # Save the last diff image for potential saving

    except Exception:
        messagebox.showerror("Capture Error", "Could not capture the move. Please try again.")
        return


def edit_last_move(initial_text=""):
    global board
    prev_move = board.peek()  # Get the last move without removing it
    move = None
    move_str = simpledialog.askstring(
        "Edit Move",
        "Enter the move in UCI format (e.g., e2e4):",
        parent=root,
        initialvalue=initial_text,
    )
    if move_str:
        try:
            board.pop()  # Remove the last move
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                update_board_display()
            else:
                messagebox.showerror("Invalid Move", "The move is not legal.")
                move = prev_move  # Revert to the previous move
        except ValueError:
            messagebox.showerror("Invalid Format", "Please enter a valid UCI format move.")
        finally:
            board.push(move or prev_move)  # Push the move back to the board


def capture_first():
    global before_image, after_image, board
    try:
        img = get_image()
        if img is None:
            raise ValueError("Could not decode image")

        warped = rectify_board(img, chessboard_corners, size=224)
        show_image(warped, color=cv2.COLOR_GRAY2RGB)

        before_image = warped
        after_image = None
        board = chess.Board()

        label.config(text="White to move. Make the move, then press 'Capture'")
        capture_btn.config(text="Capture", command=capture_move)

    except Exception as e:
        messagebox.showerror("Capture Error", str(e))
        return


def confirm_detected_board():
    global chessboard_corners

    if capture_board_with_detection.chessboard_corners is None:
        messagebox.showerror("Error", "No chessboard detected. Please try again.")
        return
    chessboard_corners = capture_board_with_detection.chessboard_corners

    show_image(None)

    label.config(text="2. Set up the pieces, then press 'Capture'")
    detect_btn.config(state=tk.DISABLED)
    confirm_btn.config(state=tk.DISABLED)
    capture_btn.config(state=tk.NORMAL)
    capture_btn.config(text="Capture", command=capture_first)


def update_board_display():
    global board_fen_label, last_move_label

    board_fen = board.board_fen()
    board_fen_label.config(text=f"FEN: {board_fen}")

    # Display the last move
    if board.move_stack:
        last_move_label.config(text=f"Last move: {board.peek().uci()}")
    else:
        last_move_label.config(text="Last move: None")


def show_image(img, color=cv2.COLOR_BGR2RGB):
    if img is None:
        image_label.configure(image="")
        return

    img_rgb = cv2.cvtColor(img, color)
    img_rgb = img_rgb.astype(np.uint8)  # Ensure the data type is uint8
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((500, 500))
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.configure(image=img_tk)
    image_label.image = img_tk


root: tk.Tk | None = None
label = detect_btn = confirm_btn = capture_btn = None
image_label = last_move_label = manual_edit_move_btn = board_fen_label = None


def main() -> None:
    """Console-script entry point: start the Tkinter phone-camera demo."""
    global device, model, NEXT_ID
    global root, label, detect_btn, confirm_btn, capture_btn
    global image_label, last_move_label, manual_edit_move_btn, board_fen_label

    if not CAMERA_URL:
        raise SystemExit(
            "Camera URL is not set. Export CHESS_DETECTOR_CAMERA_URL=http://<ip>/photo.jpg"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMoveModel(embed_dim=256, encoder_class=ENCODER_CLASS)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model_state_dict"])
    model.to(device)

    if not os.path.exists(SAVE_DIFF_PATH):
        os.makedirs(SAVE_DIFF_PATH)

    if not os.path.exists(SAVE_METADATA_PATH):
        with open(SAVE_METADATA_PATH, "w") as f:
            f.write("id,move\n")
    else:
        with open(SAVE_METADATA_PATH) as f:
            lines = f.readlines()
            NEXT_ID = int(lines[-1].split(",")[0]) + 1 if len(lines) > 1 else 0

    root = tk.Tk()
    root.title("Chessboard Detection")
    root.geometry("1024x768")
    root.configure(bg="white")

    label = tk.Label(root, text="Step 1: capture the chessboard", font=("Arial", 16), bg="white")
    label.pack(pady=10)

    detect_btn = tk.Button(root, text="🔍 Detect Chessboard", command=capture_board_with_detection)
    detect_btn.pack()

    confirm_btn = tk.Button(
        root,
        text="✅ Confirm Detection",
        command=confirm_detected_board,
        state=tk.DISABLED,
    )
    confirm_btn.pack(pady=10)

    capture_btn = tk.Button(
        root,
        text="📸 Capture",
        command=capture_board_with_detection,
        state=tk.DISABLED,
    )
    capture_btn.pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    last_move_label = tk.Label(root)
    last_move_label.pack(pady=10)

    manual_edit_move_btn = tk.Button(root, text="Edit Move", command=edit_last_move)
    manual_edit_move_btn.pack(pady=10)
    manual_edit_move_btn.config(state=tk.DISABLED)

    board_fen_label = tk.Label(root, text="", font=("Arial", 12), bg="white")
    board_fen_label.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
