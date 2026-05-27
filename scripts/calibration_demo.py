"""Standalone OpenCV chessboard calibration demo (kept for reference).

Run from the repo root with the dataset already generated under ``dataset/``.
"""

import glob

import cv2 as cv
import numpy as np

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob("./dataset/images/empty_board_*.png")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (7,7), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(5000)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)


def draw_box(img, imgpts, color, thickness=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # bottom
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]), color, thickness)
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[2]), color, thickness)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[3]), color, thickness)
    img = cv.line(img, tuple(imgpts[3]), tuple(imgpts[0]), color, thickness)

    # top
    img = cv.line(img, tuple(imgpts[4]), tuple(imgpts[5]), color, thickness)
    img = cv.line(img, tuple(imgpts[5]), tuple(imgpts[6]), color, thickness)
    img = cv.line(img, tuple(imgpts[6]), tuple(imgpts[7]), color, thickness)
    img = cv.line(img, tuple(imgpts[7]), tuple(imgpts[4]), color, thickness)

    # pillars
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[4]), color, thickness)
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[5]), color, thickness)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[6]), color, thickness)
    img = cv.line(img, tuple(imgpts[3]), tuple(imgpts[7]), color, thickness)

    return img


def rectify_board(img, corners, size=224):
    """
    Warp perspective to get a top-down view of the chessboard.

    Parameters:
        img: Input BGR or grayscale image
        corners: 4x2 array of image coordinates (top-left, top-right, bottom-right, bottom-left)
        size: Target square image size (default 224)

    Returns:
        Warped grayscale 224x224 image (float32, normalized [0,1])
    """
    # Define target square corners
    dst_pts = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )

    src_pts = np.array(corners, dtype=np.float32)

    # Compute homography
    matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, matrix, (size, size))

    # Convert to grayscale if needed
    if len(warped.shape) == 3:
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    # Normalize to [0, 1]
    warped = warped.astype(np.float32) / 255.0

    return warped


img = cv.imread("./dataset/images/b04fa05490a37470481f5bfd6b36392c.png")
h, w = img.shape[:2]

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, corners = cv.findChessboardCorners(gray, (7, 7), None)

# If found, add object points, image points (after refining them)
if ret:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

# chessboard corners
corners = np.float32([[-1, -1, 0], [7, -1, 0], [7, 7, 0], [-1, 7, 0]])

# project 3D points to image plane
imgpts, jac = cv.projectPoints(corners, rvecs, tvecs, mtx, dist)

imgpts = np.int32(imgpts).reshape(-1, 2)

img = rectify_board(img, imgpts, size=512)

# Show the result
cv.imshow("8x8 Chessboard Grid", img)
cv.waitKey(5000)

# Save the result

cv.imwrite("./8x8_chessboard_grid.png", img)
