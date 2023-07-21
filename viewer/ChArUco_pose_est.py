import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  # OpenCV version was 4.4
import cv2.aruco as aruco

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

# HoloLens camera calibration
pv_intrinsic = pd.read_csv('data/matrices/intrinsics_2023-07-20_17-01-23.csv', sep=',', header=None).values.T[:-1, :-1]
pv_extrinsic = pd.read_csv('data/matrices/extrinsics_2023-07-20_17-01-23.csv', sep=',', header=None).values.T
pv_pose = pd.read_csv('data/matrices/pv_pose_2023-07-20_17-01-23.csv', sep=',', header=None).values.T
depth_intrinsic = pd.read_csv('data/matrices/LT_intrinsics_2023-07-20_17-01-23.csv', sep=',', header=None).values.T
depth_extrinsic = pd.read_csv('data/matrices/LT_extrinsics_2023-07-20_17-01-23.csv', sep=',', header=None).values.T
depth_pose = pd.read_csv('data/matrices/lt_pose_2023-07-20_17-01-23.csv', sep=',', header=None).values.T

# 1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
# Can convert to a class with methods later on.

# ChArUco board specs (similar to calib.io)
squaresX = 6  # no. of squares in X-direction
squaresY = 6  # no. of squares in Y-direction
# board_width = 0.20  # in m
# board_height = 0.20  # in m
square_length = 0.030  # in m
marker_length = 0.023  # in m
pixels_num_height = 600
pixels_num_width = 600

# Defining ArUco dictionary for markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
# aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]

# Create ChArUco board and view it
board = aruco.CharucoBoard_create(
    squaresX, squaresY, square_length, marker_length, aruco_dict)
imboard = board.draw((pixels_num_height, pixels_num_width), 10, 1)

# 2. DETECT CHARUCO BOARD CORNERS
# Convert to function

# Detect the markers in the image
img = cv2.imread('data/rgb_images/20230720_170236_HoloLens.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Original image', cv2.resize(img, (960, 540)))
# cv2.imshow('Gray image', cv2.resize(gray, (960, 540)))
# cv2.waitKey(0)

# Check for AruCo dictionary
aruco_params = aruco.DetectorParameters_create()

# Detect the markers in the image
img_corners, ids, rejected = aruco.detectMarkers(
    gray, aruco_dict, parameters=aruco_params)

#   Interpolate position of ChArUco board corners.
_, charuco_corners, charuco_id = aruco.interpolateCornersCharuco(
    img_corners, ids, gray, board, cameraMatrix=None, distCoeffs=None)

# Draw detected markers on the image
debug_img = aruco.drawDetectedMarkers(img, img_corners, ids)

# Draw detected corners on markers
aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_id, (0, 0, 255))
plt.imshow(debug_img)
plt.show()

# Print the IDs and corner coordinates of the detected markers
if ids is not None:
    for i in range(len(ids)):
        # Calculate centre point
        centre = np.mean(img_corners[i][0], axis=0)

        # Draw a circle at the center point
        cv2.circle(gray, tuple(centre.astype(int)), 3, (0, 255, 0), -1)

    # Display the image with detected markers
    cv2.namedWindow("Detected markers", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected markers", 1000, 1000)
    cv2.imshow("Detected markers", debug_img)
    # cv2.imshow("ChArUco board", imboard)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Remove ArUco corners not detected in the image.
board_corners = np.array(board.objPoints)
board_ids = board.ids.tolist()

# 3. OBTAIN POSE (RVEC AND TVEC) OF BOARD WITH RESPECT TO RGB CAMERA.
outcome, rvecs, tvecs = aruco.estimatePoseCharucoBoard(
    charucoCorners=charuco_corners, charucoIds=charuco_id,
    board=board, cameraMatrix=pv_intrinsic, distCoeffs=None,
    rvec=None, tvec=None)

# Get rotation and pose (rotation + translation) matrices
rotation_mat, _ = cv2.Rodrigues(rvecs)
T_bc = np.hstack((rotation_mat, tvecs))
projection = np.matmul(pv_intrinsic, T_bc)

# Get corresponding image points of rigid base corners from measured positions in board coordinates.
padding = 10
board_points = np.array([[padding+77., 76.-padding, -3.], [padding+77., 106.-padding, -3.],
    [padding+105., 106.-padding, -3.], [padding+105., 76.-padding, -3.]]) / 1000.
img_rigid_points = []
img_points = []

for idx, corner in enumerate(board_points):
    transformed_point = np.matmul(projection, np.hstack((corner, 1.)))
    img_points.append(transformed_point / transformed_point[-1])
    img_rigid_points.append(transformed_point)
    cv2.circle(img, tuple(img_points[idx][:-1].astype(int)),
        radius=10, color=(0, 0, 255), thickness=-1)

# Display pose of board with respect to RGB camera
cv2.drawFrameAxes(debug_img, cameraMatrix=pv_intrinsic, distCoeffs=None,
                  rvec=rvecs, tvec=tvecs, length=0.02, thickness=10)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 1000, 1000)
cv2.imshow("Board pose", debug_img)
cv2.waitKey(0)

# Finding colour to depth transformation
T_cd = np.matmul(depth_extrinsic, np.linalg.inv(pv_extrinsic))

# Going straight from board to depth
T_bd = np.matmul(T_cd, np.vstack((T_bc,[0., 0., 0., 1.])))
depth = np.matmul(T_bd[:-1, :], board_points[0])
print(depth)