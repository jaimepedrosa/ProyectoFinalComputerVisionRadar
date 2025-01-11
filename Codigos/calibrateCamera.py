from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import matplotlib.pyplot as plt

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    num_rows, num_columns = chessboard_shape
    objective_points = []

    for i in range(num_rows):
        for j in range(num_columns):
            objective_points.append([j*dx, i*dy, 0])

    objective_points = np.array(objective_points, dtype=np.float32)            

    return objective_points

def calibrateCamera():
    imgs_path = glob.glob('../assets/calibracion/*jpg')
    imgs = load_images(imgs_path)

    corners = []
    for img in imgs:
        corner = cv2.findChessboardCorners(img, (8,6))
        corners.append(corner)
    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    imgs_gray = []
    for img in imgs:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs_gray.append(img_gray)
        
    corners_refined = [cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)
    for i in range(len(imgs_copy)):
        cv2.drawChessboardCorners(imgs_copy[i],patternSize=(8,6), corners=corners[i][1], patternWasFound = corners[i][0])

    chessboard_points = get_chessboard_points((8, 6), 30, 30)

    chessboard_points_list = [chessboard_points for j in range(18)]
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    img_sze = (imgs[0].shape[1], imgs[0].shape[0]) 
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points_list, valid_corners, img_sze, None, None)

    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    return rms, intrinsics, dist_coeffs



if __name__ == "__main__":

    rms, intrinsics, dist_coeffs = calibrateCamera()

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)
