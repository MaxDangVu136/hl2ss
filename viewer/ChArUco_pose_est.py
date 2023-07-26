import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  # OpenCV version was 4.4

"""
- We don't want the marker corners for pose estimation. We use them to interpolate the position of 
    checkerboard corners which are more accurate to detect.
- Not a good idea to use cv2.calibratecam for camera calibration with only one image. Need multiple images in different 
    orientations. In my case, this is already done by HoloLens.
- OpenCV functions are for C++ libraries. Functions like estimatePoseCharucoBoard are implemented to take inputs 
    to use as placeholders for eventual outputs (e.g. rvec and tvec). In this case, can simply add "None" as input.
- Easier to measure points in board coordinates, then find their corresponding position in RGB space 
    than other way around.
"""

# Image time
im_time = '2023-07-25_14-31-29'

# HoloLens camera calibration
pv_intrinsic = pd.read_csv('data/matrices/intrinsics_{}.csv'.format(im_time),
                           sep=',', header=None).values.T[:-1, :-1]
pv_distort = np.array([0., 0., 0., 0., 0.])
pv_extrinsic = pd.read_csv('data/matrices/extrinsics_{}.csv'.format(im_time),
                           sep=',', header=None).values.T
pv_pose = pd.read_csv('data/matrices/pv_pose_{}.csv'.format(im_time),
                      sep=',', header=None).values.T
depth_intrinsic = pd.read_csv('data/matrices/LT_intrinsics_{}.csv'.format(im_time),
                              sep=',', header=None).values.T
depth_extrinsic = pd.read_csv('data/matrices/LT_extrinsics_{}.csv'.format(im_time),
                              sep=',', header=None).values.T
depth_pose = pd.read_csv('data/matrices/lt_pose_{}.csv'.format(im_time),
                         sep=',', header=None).values.T

# 1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
# Can convert to a class with methods later on.

# ChArUco board specs (similar to calib.io)
Board16x12 = {
    'squares_x': 16,
    'squares_y': 12,
    'square_length': 0.016,  # in m
    'marker_length': 0.012,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}

# For case of 12 x 9 squares.
Board12x9 = {
    'squares_x': 12,
    'squares_y': 9,
    'square_length': 0.022,  # in m
    'marker_length': 0.017,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}

# For case of 9 by 7 squares
Board9x7 = {
    'squares_x': 9,
    'squares_y': 7,
    'square_length': 0.027,  # in m
    'marker_length': 0.021,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}

# For case of 6 by 6 squares
Board6x6 = {
    'squares_x': 6,
    'squares_y': 6,
    'square_length': 0.030,  # in m
    'marker_length': 0.023,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}

# Create ChArUco board and view it
dim = Board16x12
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(dim['aruco_dict'])
board = cv2.aruco.CharucoBoard_create(
    dim['squares_x'], dim['squares_y'],
    dim['square_length'], dim['marker_length'], aruco_dict)

# 2. DETECT CHARUCO BOARD CORNERS
# Convert to function

# Detect the markers in the image
img = cv2.imread('data/rgb_images/20230725_143135_HoloLens.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', cv2.resize(gray, (960, 540)))
cv2.waitKey(0)

# Detect the markers in the image
img_corners, ids, rejected = cv2.aruco.detectMarkers(
    gray, aruco_dict, parameters=aruco_params)

#   Interpolate position of ChArUco board corners.
_, charuco_corners, charuco_id = cv2.aruco.interpolateCornersCharuco(
    img_corners, ids, gray, board, cameraMatrix=None, distCoeffs=None)

# Draw detected markers on the image
debug_img = cv2.aruco.drawDetectedMarkers(img, img_corners, ids)

# Draw detected corners on markers
cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_id, (0, 0, 255))
plt.imshow(debug_img)
plt.show()

# Print the IDs and corner coordinates of the detected markers
if ids is not None:
    for i in range(len(ids)):
        # Calculate centre point
        centre = np.mean(img_corners[i][0], axis=0)

        # Draw a circle at the center point
        cv2.circle(gray, tuple(centre.astype(int)), 3, (0, 255, 0), -1)

# 3. OBTAIN POSE (RVEC AND TVEC) OF BOARD WITH RESPECT TO RGB CAMERA.
outcome, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
    charucoCorners=charuco_corners, charucoIds=charuco_id,
    board=board, cameraMatrix=pv_intrinsic, distCoeffs=None,
    rvec=None, tvec=None)

# Get rotation and pose (rotation + translation) matrices
rotation_mat, _ = cv2.Rodrigues(rvecs)
T_bc = np.hstack((rotation_mat, tvecs))
project_bc = np.matmul(pv_intrinsic, T_bc)

# Get corresponding image points of rigid base corners from measured positions in board coordinates.
centre = np.array([dim['squares_x'] * dim['square_length'],
                   dim['squares_y'] * dim['square_length']]) / 2
cube = 0.03
board_points = np.array([[centre[0] - cube / 2, centre[1] - cube / 2, -0.003],
                         [centre[0] - cube / 2, centre[1] + cube / 2, -0.003],
                         [centre[0] + cube / 2, centre[1] + cube / 2, -0.003],
                         [centre[0] + cube / 2, centre[1] - cube / 2, -0.003]])
img_rigid_points = []
img_points = []

for idx, corner in enumerate(board_points):
    transformed_point = np.matmul(project_bc, np.hstack((corner, 1.)))
    img_points.append(transformed_point / transformed_point[-1])
    img_rigid_points.append(transformed_point)
    cv2.circle(img, tuple(img_points[idx][:-1].astype(int)),
               radius=5, color=(255, 255, 0), thickness=-1)

# Display pose of board with respect to RGB camera
cv2.drawFrameAxes(debug_img, cameraMatrix=pv_intrinsic, distCoeffs=pv_distort,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 960, 540)
cv2.imshow("Board pose", debug_img)
cv2.waitKey(0)

# 4. BOARD TO DEPTH SPACE

# Finding colour to depth transformation
T_cd = np.matmul(depth_extrinsic, np.linalg.inv(pv_extrinsic))

# Going straight from board to depth
T_bd = np.matmul(T_cd, np.vstack((T_bc, [0., 0., 0., 1.])))
depth = np.matmul(T_bd[:-1, :], np.hstack((board_points,
                                           np.ones(shape=(len(board_points), 1)))).T)

# 5. TRANSFORM DEPTH DATA TO SHARED COORDINATE SYSTEM (BOARD COORDINATE SPACE)
depth_raw = pd.read_csv('data/points/lt_{}.csv'.format(im_time), sep=',', header=None).values
depth_data = np.array([np.hstack((np.fromstring(element[1:-1], sep=' '), 1.))
                       for px in depth_raw for element in px]).reshape(-1, depth_raw.shape[1], 4)
depth_to_board = np.zeros((depth_raw.shape[0], depth_raw.shape[1], 3))
depth_to_board = np.array([np.matmul(np.linalg.inv(T_bd)[:-1, :], depth_data[px_x, px_y, :].T).T
                           for px_y in range(depth_data.shape[1]) for px_x in range(depth_data.shape[0])])\
                            .reshape(-1, depth_data.shape[1], 3)
