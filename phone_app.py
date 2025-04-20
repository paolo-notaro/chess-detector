import tkinter as tk
from tkinter import messagebox
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
    except:
        messagebox.showerror("Capture Error", "Could not capture the move. Please try again.")
        return

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
root.geometry("600x700")
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

board_fen_label = tk.Label(root, text="", font=("Arial", 12), bg="white")
board_fen_label.pack(pady=10)

root.mainloop()
