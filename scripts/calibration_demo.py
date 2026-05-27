"""Standalone OpenCV chessboard calibration demo (kept for reference).

Run from the repo root with the dataset already generated under
``$CHESS_DETECTOR_DATA_DIR`` (default ``./dataset``). The script:

1. Loads every ``empty_board_*.png`` and recovers the camera intrinsics.
2. Picks one rendered scene, runs PnP, projects the board corners back into
   the image, and warps it to a 512x512 top-down view.

Aborts cleanly if no calibration images or scene image are present, instead
of crashing with ``UnboundLocalError``/``NameError``.
"""

from __future__ import annotations

import glob
import sys

import cv2 as cv
import numpy as np

from chess_detector.data import paths

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SCENE_IMAGE_NAME = "b04fa05490a37470481f5bfd6b36392c.png"
OUTPUT_PATH = "8x8_chessboard_grid.png"


def _make_object_points() -> np.ndarray:
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    return objp


def rectify_board(img: np.ndarray, corners: np.ndarray, size: int = 224) -> np.ndarray:
    """Warp the perspective of ``img`` to a top-down ``size``x``size`` board."""
    dst_pts = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    src_pts = np.array(corners, dtype=np.float32)
    matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, matrix, (size, size))
    if len(warped.shape) == 3:
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    return warped.astype(np.float32) / 255.0


def _calibrate_from_empty_boards(objp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    last_gray: np.ndarray | None = None

    for fname in glob.glob(paths.empty_boards_glob()):
        img = cv.imread(fname)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
        if not ret:
            continue
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        imgpoints.append(corners2)
        last_gray = gray

    if not objpoints or last_gray is None:
        raise FileNotFoundError(
            f"No usable calibration images at {paths.empty_boards_glob()!r}; "
            "run chess-detector-gen-dataset first."
        )

    _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, last_gray.shape[::-1], None, None)
    return objp, mtx, dist


def main() -> int:
    """Console entry point: calibrate, rectify a sample scene, and save the result."""
    objp = _make_object_points()
    try:
        objp, mtx, dist = _calibrate_from_empty_boards(objp)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    scene_path = paths.images_dir() / SCENE_IMAGE_NAME
    img = cv.imread(str(scene_path))
    if img is None:
        print(f"Scene image not found: {scene_path}", file=sys.stderr)
        return 1

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
    if not ret:
        print(f"Could not detect a chessboard in {scene_path}", file=sys.stderr)
        return 1
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    _, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    board_corners = np.array([[-1, -1, 0], [7, -1, 0], [7, 7, 0], [-1, 7, 0]], dtype=np.float32)
    imgpts, _ = cv.projectPoints(board_corners, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    rectified = rectify_board(img, imgpts, size=512)
    cv.imshow("8x8 Chessboard Grid", rectified)
    cv.waitKey(5000)
    cv.imwrite(OUTPUT_PATH, rectified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
