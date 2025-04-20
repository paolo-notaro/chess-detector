import glob
import json
import os

import cv2 as cv
import numpy as np

# Validate set ratio
VALIDATE_SET_RATIO = 0.2


# chessboard corners
OBJ_SPACE_CORNERS = np.float32([
    [-1, -1, 0],
    [7, -1, 0],
    [7, 7, 0],
    [-1, 7, 0]
])

def calibrate_camera_from_path_match(path_match):

    """ Return the image points of the chessboard corners, calibrated from empty board images from the path match. """

    images = glob.glob(path_match)

    return calibrate_camera_from_images([cv.imread(fname) for fname in images])

def calibrate_camera_from_images(img_list: list):
            # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for img in img_list:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (7,7), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(5000)
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

    imgpts, jac = cv.projectPoints(OBJ_SPACE_CORNERS, rvecs, tvecs, mtx, dist)

    imgpts = np.int32(imgpts).reshape(-1,2)

    return imgpts

def reorder_points(pts: np.ndarray) -> np.ndarray:
    """
    Reorder 4 points to top-left, top-right, bottom-right, bottom-left.

    Parameters:
        pts (np.ndarray): Array of shape (4, 2)

    Returns:
        np.ndarray: Reordered array of shape (4, 2)
    """
    if pts.shape != (4, 2):
        raise ValueError("Input must be a (4, 2) array of points")

    pts = pts.astype(np.float32)  # for safety

    # Sum and diff of points
    s = pts.sum(axis=1)           # x + y
    d = np.diff(pts, axis=1).flatten()  # x - y

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # Top-left
    ordered[2] = pts[np.argmax(s)]      # Bottom-right
    ordered[1] = pts[np.argmin(d)]      # Top-right
    ordered[3] = pts[np.argmax(d)]      # Bottom-left

    return ordered

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
    dst_pts = np.array([
        [0, 0],
        [size-1, 0],
        [size-1, size-1],
        [0, size-1]
    ], dtype=np.float32)

    src_pts = np.array(corners, dtype=np.float32)

    ordered_src_pts = reorder_points(src_pts)

    # Compute homography
    matrix = cv.getPerspectiveTransform(ordered_src_pts, dst_pts)
    warped = cv.warpPerspective(img, matrix, (size, size))

    # Convert to grayscale if needed
    if len(warped.shape) == 3:
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    # Convert to float32
    warped = warped.astype(np.float32)

    return warped

def gen_diff(before_img, after_img, binary=False, binary_threshold=30):
    """
    Generate a difference image between before and after images.

    Parameters:
        before_img: Before image (grayscale or BGR)
        after_img: After image (grayscale or BGR)"
    """
    diff_img = cv.absdiff(before_img, after_img)

    if binary:
        _, diff_img_binary = cv.threshold(diff_img, binary_threshold, 255, cv.THRESH_BINARY)
        return diff_img_binary
    else:
        return diff_img

def process_image(frompath, chessboard_corners):
    img = cv.imread(frompath)

    img = rectify_board(img, chessboard_corners)

    return img

def save_image(img, topath):
    cv.imwrite(topath, img)
    print(f"Image {topath} saved.")