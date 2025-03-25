import numpy as np
import cv2 as cv
import json
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# chessboard corners
OBJ_SPACE_CORNERS = np.float32([
    [-1, -1, 0],
    [7, -1, 0],
    [7, 7, 0],
    [-1, 7, 0]
])

def calibrate_camera():
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./images/empty_board_*.png')

    for fname in images:
        img = cv.imread(fname)
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

    return ret, mtx, dist, rvecs, tvecs

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

    # Compute homography
    matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, matrix, (size, size))

    # Convert to grayscale if needed
    if len(warped.shape) == 3:
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    # Normalize to [0, 1]
    warped = warped.astype(np.float32) / 255.0

    return warped

if __name__ == '__main__':
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    os.makedirs('preprocessed', exist_ok=True)
    
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()

    for board_id in metadata['boards']:
        if os.path.exists(f'preprocessed/{board_id}.png'):
            print(f'Board {board_id} already exists.')
            continue
        
        img = cv.imread(f'images/{board_id}.png')

        imgpts, jac = cv.projectPoints(OBJ_SPACE_CORNERS, rvecs, tvecs, mtx, dist)

        imgpts = np.int32(imgpts).reshape(-1,2)

        img = rectify_board(img, imgpts)

        cv.imwrite(f'preprocessed/{board_id}.png', img * 255.0)

        print(f'Board {board_id} processed.')
