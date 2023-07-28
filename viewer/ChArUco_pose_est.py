import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  # OpenCV version was 4.4
import open3d as o3d

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

# 0. LOADING RELEVANT IMAGES
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

#  Depth and RGB images
depth_data_path = 'data/points/depth_2023-07-25_14-31-29.csv'
depth_map = pd.read_csv(depth_data_path, sep=',', header=None).values
img = cv2.imread('data/rgb_images/20230725_143135_HoloLens.jpg')

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
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

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
pose_bc = np.hstack((rotation_mat, tvecs))
T_bc = pv_intrinsic @ pose_bc

# Get corresponding image points of rigid base corners from measured positions in board coordinates.
centre = np.array([dim['squares_x'] * dim['square_length'],
                   dim['squares_y'] * dim['square_length']]) / 2
cube = 0.03
board_points = np.array([[centre[0] - cube / 2, centre[1] - cube / 2, -0.003],
                         [centre[0] - cube / 2, centre[1] + cube / 2, -0.003],
                         [centre[0] + cube / 2, centre[1] + cube / 2, -0.003],
                         [centre[0] + cube / 2, centre[1] - cube / 2, -0.003]])
board_points_h = np.hstack((board_points, np.ones(shape=(board_points.shape[0], 1)))).T

# Board to colour coordinates
img_points_h = T_bc @ board_points_h
img_points = (img_points_h[:-1, :]/img_points_h[-1, :]).T

for idx in range(img_points.shape[0]):
    cv2.circle(img, tuple(img_points[idx, :].astype(int)),
               radius=5, color=(255, 255, 0), thickness=-1)

# Display pose of board with respect to RGB camera
cv2.drawFrameAxes(debug_img, cameraMatrix=pv_intrinsic, distCoeffs=pv_distort,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 960, 540)
cv2.imshow("Board pose", debug_img)
cv2.waitKey(0)

# 4. BOARD TO DEPTH SPACE

# Collect all relevant transformations
T_hc = pv_extrinsic
T_hd = depth_extrinsic
T_cd = T_hd @ np.linalg.inv(T_hc)
T_bc = np.vstack((T_bc, [0., 0., 0., 1.]))
T_bd = T_hd @ np.linalg.inv(T_hc) @ T_bc

# Board coordinates to depth coordinates
corner_depth_homogenous = T_bd @ board_points_h
corner_depth = (corner_depth_homogenous[:-1, :]/corner_depth_homogenous[-1, :]).T

# Colour to depth coordinates
img_points_h_3d = np.vstack((img_points_h, np.ones(shape=(1, board_points.shape[0]))))
colour_depth_homogenous = T_cd @ img_points_h_3d
colour_depth = (colour_depth_homogenous[:-1, :]/colour_depth_homogenous[-1, :]).T

# Depth to board coordinates
corner_depth_board = (np.linalg.inv(T_bd) @ corner_depth_homogenous).T[:, :-1]
colour_depth_board = (np.linalg.inv(T_bd) @ colour_depth_homogenous).T[:, :-1]


# 5. TRANSFORM DEPTH DATA TO SHARED COORDINATE SYSTEM (BOARD COORDINATE SPACE)

# Depth point cloud
pcd = o3d.io.read_point_cloud('data/point_clouds/binary/'
                              'depth_point-cloud_{}.ply'.format(im_time))
depth_pcd = np.asarray(pcd.points)
depth_pcd_h = np.hstack((depth_pcd, np.ones(shape=(depth_pcd.shape[0], 1)))).T
depth_to_board_h = np.linalg.inv(T_bd) @ depth_pcd_h
depth_to_board = (depth_to_board_h[:-1, :]/depth_pcd_h[-1, :]).T

# Using depth map
U, V = np.meshgrid(np.array([list(range(1, depth_map.shape[0] + 1))]),
                   np.array([list(range(1, depth_map.shape[1] + 1))]))
print(U.reshape(-1, 1))
print(V.reshape(-1, 1))
depth_img = np.array([U.reshape(-1, 1), V.reshape(-1, 1), depth_map.reshape(-1, 1)]).T.reshape(-1, 3)
depth_img_h = np.hstack((depth_img, np.ones(shape=(depth_img.shape[0], 1)))).T
depth_map_to_board = np.linalg.inv(T_bd) @ depth_img_h
depth_map_to_board =(depth_map_to_board[:-1, :]/depth_map_to_board[-1, :]).T

# Compare depth to board transformed data vs original depth data
fig = plt.figure()

ax_db = fig.add_subplot(1, 2, 1, projection='3d')
scatter_db = ax_db.scatter(depth_to_board[:, 0], depth_to_board[:, 1],
                           depth_to_board[:, 2], c=depth_to_board[:, 2], cmap='magma')
cbar_db = fig.colorbar(scatter_db, ax=ax_db)
cbar_db.set_label('depth estimate')
ax_db.set_xlabel('x')
ax_db.set_ylabel('y')
ax_db.set_zlabel('z')
ax_db.set_title('Depth map to board transformation')

ax_d = fig.add_subplot(1, 2, 2, projection='3d')
scatter_d = ax_d.scatter(depth_pcd[:, 0], depth_pcd[:, 1],
                         depth_pcd[:, 2], c=depth_pcd[:, 2], cmap='magma')
cbar_d = fig.colorbar(scatter_d, ax=ax_d)
cbar_d.set_label('depth estimate')
ax_d.set_xlabel('x')
ax_d.set_ylabel('y')
ax_d.set_zlabel('z')
ax_d.set_title('Depth data')

plt.show()


## 6. COMPUTE EULER ANGLE FROM T_BD
def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        order(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'xyz')
print(t1, t2, t3)
t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'yxz')
print(t1, t2, t3)

t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'xzy')
print(t1, t2, t3)
t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'zxy')
print(t1, t2, t3)

t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'yzx')
print(t1, t2, t3)
t1, t2, t3 = rotation_angles(np.linalg.inv(T_bd)[:-1, :-1], 'zyx')
print(t1, t2, t3)
