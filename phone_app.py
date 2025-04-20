import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import requests
import cv2
import numpy as np
from dataset.postprocessing import calibrate_camera_from_images, rectify_board
import chess

CAMERA_URL = "http://192.168.10.61:8080/photo.jpg"  # Update IP as needed

CAPTURE_CALIB_IMGS = 5  # Number of images to capture for calibration

chessboard_corners = None
before_image = None
after_image = None
board = None

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

        cv2.polylines(img, [corners.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        capture_board_with_detection.annotated_img = img
        show_image(img)

        confirm_btn.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Detection Error", str(e))

def capture_move():
    global before_image, after_image, board
    try:
        img = get_image()
        if img is None:
            raise ValueError("Could not decode image")
        
        after_image = rectify_board(img, chessboard_corners, size=224)
        show_image(after_image)

        # TODO: Retrieve move prediction from model

        pred_uci = "e2e4"  # Placeholder for predicted move
        move = chess.Move.from_uci(pred_uci)

        while not move in board.legal_moves:
            move_str = simpledialog.askstring("Edit Move", "The move is illegal. Enter the move in UCI format (e.g., e2e4):", parent=root, initialvalue=pred_uci)
            if move_str:
                move = chess.Move.from_uci(move_str)

        board.push(move)
        last_move_label.config(text=f"Last move: {move.uci()}")
        board_fen_label.config(text=f"FEN: {board.fen()}")
        manual_edit_move_btn.config(state=tk.NORMAL)
        label.config(text=f"Move captured. Press 'Edit Move' to modify or 'Capture' to capture again. {board.turn == chess.WHITE and 'White' or 'Black'} to move.")
        manual_edit_move_btn.config(state=tk.NORMAL)
        capture_btn.config(text="Capture", command=capture_move)

    except:
        messagebox.showerror("Capture Error", "Could not capture the move. Please try again.")
        return
    
def edit_last_move(initial_text=""):
    global board
    board.pop()  # Remove the last move
    last_move_label.config(text="Last move: None")
    move_str = simpledialog.askstring("Edit Move", "Enter the move in UCI format (e.g., e2e4):", parent=root, initialvalue=initial_text)
    if move_str:
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
                last_move_label.config(text=f"Last move: {move.uci()}")
                board_fen_label.config(text=f"FEN: {board.fen()}")
            else:
                messagebox.showerror("Invalid Move", "The move is not legal.")
        except ValueError:
            messagebox.showerror("Invalid Format", "Please enter a valid UCI format move.")

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

def show_image(img, color=cv2.COLOR_BGR2RGB):
    if img is None:
        image_label.configure(image='')
        return
    
    img_rgb = cv2.cvtColor(img, color)
    img_rgb = img_rgb.astype(np.uint8)  # Ensure the data type is uint8
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((500, 500))
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

# ---- UI Setup ----
root = tk.Tk()
root.title("Chessboard Detection")
root.geometry("800x600")
root.configure(bg="white")

label = tk.Label(root, text="Step 1: capture the chessboard", font=("Arial", 16), bg="white")
label.pack(pady=10)

detect_btn = tk.Button(root, text="🔍 Detect Chessboard", command=capture_board_with_detection)
detect_btn.pack()

confirm_btn = tk.Button(root, text="✅ Confirm Detection", command=confirm_detected_board, state=tk.DISABLED)
confirm_btn.pack(pady=10)

capture_btn = tk.Button(root, text="📸 Capture", command=capture_board_with_detection, state=tk.DISABLED)
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
