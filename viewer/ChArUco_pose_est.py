import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2  # OpenCV version was 4.4
import open3d as o3d


# FUNCTIONS
def load_hololens_matrices_at_time(time):
    """
    This function loads the intrinsic and extrinsic matrices of the HoloLens 2's PV (RGB)
    and depth (LT) cameras for an image captured at a specified time using the HoloLens 2.

    Parameters
    ---------
    time: str
        Time at which an RGB and depth image has been taken in the format (year-month-day_24htime).

    Returns
    -------
    pv_intrinsic: array
        4x4 calibration matrix of the RGB camera, containing the camera's focal length and principal point.
    pv_distort: array
        1x5 matrix containing the RGB camera lens' radial and tangential distortion.
    pv_extrinsic: array
        4x4 extrinsic matrix that transforms points from the HoloLens reference coordinate frame
        to the RGB camera frame.
    pv_pose: array
        4x4 matrix defining the pose of the RGB camera with respect to the HoloLens coordinate system.
    depth_intrinsic: array
        4x4 calibration matrix of the depth camera.
    depth_extrinsic: array
        4x4 extrinsic matrix that transforms points from the HoloLens reference coordinate frame
        to the depth camera frame.
    depth_pose: array
        4x4 matrix defining the pose of the depth camera with respect to the HoloLens coordinate system.
    """

    pv_intrinsic = pd.read_csv('data/matrices/intrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T[:-1, :-1]
    pv_distort = np.array([0., 0., 0., 0., 0.])
    pv_extrinsic = pd.read_csv('data/matrices/extrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T
    pv_pose = pd.read_csv('data/matrices/pv_pose_{}.csv'.format(time),
                          sep=',', header=None).values.T
    lt_intrinsic = pd.read_csv('data/matrices/LT_intrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T
    lt_extrinsic = pd.read_csv('data/matrices/LT_extrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T
    lt_pose = pd.read_csv('data/matrices/lt_pose_{}.csv'.format(time),
                          sep=',', header=None).values.T

    return pv_intrinsic, pv_distort, pv_extrinsic, pv_pose, lt_intrinsic, lt_extrinsic, lt_pose


def get_depth_map(time):
    """
    Loads depth map from a depth image taken at a specified time.

    Parameters
    ---------
    time: str
        Time at which an RGB and depth image has been taken in the format (year-month-day_24htime).

    Return
    ------
    depth_map: array
        Estimated depth of a point at a given pixel in the depth camera frame.
    """

    path = 'data/points/depth_{}.csv'.format(time)
    depth_values = pd.read_csv(path, sep=',', header=None).values

    return depth_values


def generate_charuco_board(specs):
    """
    This function generates a ChArUco board (chessboard + ArUco markers) based on specifications defined in
    a prescribed dictionary.

    Parameters
    ---------
    specs: dictionary
        Contains the specifications to create a ChArUco board of m squares by n squares.

    Returns
    -------
    board: aruco_CharucoBoard
        Charuco board object containing coordinates of Chessboard corners, marker IDs and corner points of markers
        in board coordinate space.
    aruco_params: aruco_DetectorParameters
        ArUco parameters to enable ArUco marker detection.
    aruco_dict: dictionary
        Dictionary for a set of unique ArUco markers of the same size.
    """

    aruco_param = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.getPredefinedDictionary(specs['aruco_dict'])
    charuco_board = cv2.aruco.CharucoBoard_create(
        specs['squares_x'], specs['squares_y'],
        specs['square_length'], specs['marker_length'], aruco_dict)

    return charuco_board, aruco_param, aruco_dict


def detect_display_markers(charuco_board, img, aruco_dict, aruco_param, cam_mtx, dist_coeffs):
    """
    This function detects and displays ChArUco markers found in an RGB image frame.

    Parameters
    ---------
    charuco_board: aruco_CharucoBoard
        Charuco board object containing coordinates of Chessboard corners, marker IDs and
        corner points of markers in board coordinate space.
    img: array
        m x n x 3 RGB image.
    aruco_dict: dictionary
        Dictionary for a set of unique ArUco markers of the same size.
    aruco_param: aruco_DetectorParameters
        ArUco parameters to enable ArUco marker detection.
    cam_mtx: array
        4x4 calibration matrix of the RGB camera, containing the camera's focal length and principal point.
    dist_coeffs: array
        1x5 matrix containing the RGB camera lens' radial and tangential distortion.

    Returns
    -------
    debug_img: array
        m x n x 3 RGB image overlaid with interpolated ChArUco corners and marker ids.
    img_corners: list
        Position of ArUco marker corners in the RGB image.
    ids: array
        Ids of detected ArUco markers in the image.
    chessboard_corners: array
        Interpolated position of chessboard corners in the RGB image.
    chessboard_id: array
        Ids of detected chessboard corners in the image.
    """

    # Convert img from rgb to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title('RGB image with Charuco board')
    plt.show()

    # Detect the markers in the image
    (img_corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_param)

    # Interpolate position of ChArUco board corners
    _, chessboard_corners, chessboard_id = cv2.aruco.interpolateCornersCharuco(
        img_corners, ids, gray, charuco_board, cam_mtx, dist_coeffs)

    # Draw detected markers on the image
    debug_img = cv2.aruco.drawDetectedMarkers(img, img_corners, ids)

    return gray, debug_img, img_corners, ids, chessboard_corners, chessboard_id


def CharucoBoard_to_HoloLensRGB(rvec, tvec, pv_intrinsic):
    """
    This function computes the transformation matrix from ChArUco board coordinates to the HoloLens RGB camera space.

    Parameters
    ----------
    rvec: array
        1x3 rotation vector showing rotation of object about the X, Y, Z axes with respect to the RGB camera.
    tvec: array
        1x3 translation vector from centre of RGB camera to object.
    pv_intrinsic: array
        4x4 calibration matrix of the RGB camera, containing the camera's focal length and principal point.

    Return
    ------
    t_bc: array
        4x4 transformation matrix from board coordinates to RGB camera space.
    """

    rotation_mat, _ = cv2.Rodrigues(rvec)
    pose_bc = np.hstack((rotation_mat, tvec))
    t_bc = pv_intrinsic @ pose_bc

    return t_bc


def homogeneous_coordinates(points, stack):
    """
    This function adds n columns of 1's to a 2D array to perform transformation from space X to space Y.

    Parameters
    ---------
    points: array
        n x 3 array of coordinates
    stack: str
        'column' means stacking a new array to existing array horizontally (i.e. adding columns)
        'row' means stacking a new array to existing array vertically (i.e. adding rows)

    Returns
    -------
    np.hstack((points, np.ones(shape=(points.shape[0], 1)))).T
        n x 4 array of coordinates, where the last column is only 1's
    """

    if stack == 'column':
        return np.hstack((points, np.ones(shape=(points.shape[0], 1))))

    elif stack == 'row':
        return np.vstack((points, np.ones(shape=(1, points.shape[1]))))


def rigid_base_corners_on_board(board_spec, cube_size):
    """
    This function computes the position of the rigid base corners of the cantilever beam in board coordinates.

    Parameters
    ---------
    board_spec: dictionary
        Contains the specifications to create a ChArUco board of m squares by n squares.

    cube_size: float
        Dimensions of the square cross-section of the cantilever beam attached to the rigid base.

    Return
    ------
    homogeneous_coordinates(board_points)
        4 x n array of rigid base coordinates in board space, where the last column is only 1's.
    """

    centre_pt = np.array([board_spec['squares_x'] * board_spec['square_length'],
                          board_spec['squares_y'] * board_spec['square_length']]) / 2
    board_points = np.array([[centre_pt[0] - cube_size / 2, centre_pt[1] - cube_size / 2, -0.003],
                             [centre_pt[0] - cube_size / 2, centre_pt[1] + cube_size / 2, -0.003],
                             [centre_pt[0] + cube_size / 2, centre_pt[1] + cube_size / 2, -0.003],
                             [centre_pt[0] + cube_size / 2, centre_pt[1] - cube_size / 2, -0.003]])

    return homogeneous_coordinates(board_points, 'column').T


def transform_from_X_to_Y(t_mtx, x_h):
    """
    This function transforms a set of points from X coordinate space to Y coordinate space.

    Parameters
    ---------
    t_mtx: array
        4 x 4 transformation matrix from space X to Y.
    x_h: array
        4 x n array of homogeneous coordinates in X space.

    Returns
    -------
    y: array
        n x 3 array of coordinates in Y space.
    y_h: array
        4 x n array of homogeneous coordinates in Y space.
    """

    y_h = t_mtx @ x_h
    y = (y_h[:-1, :] / y_h[-1, :]).T

    return y, y_h


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
        theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
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

    return theta1, theta2, theta3


# ChArUco board specs (m squares x n squares)
Board16x12 = {
    'squares_x': 16,
    'squares_y': 12,
    'square_length': 0.016,  # in m
    'marker_length': 0.012,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}
Board12x9 = {
    'squares_x': 12,
    'squares_y': 9,
    'square_length': 0.022,  # in m
    'marker_length': 0.017,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}
Board9x7 = {
    'squares_x': 9,
    'squares_y': 7,
    'square_length': 0.027,  # in m
    'marker_length': 0.021,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}
Board6x6 = {
    'squares_x': 6,
    'squares_y': 6,
    'square_length': 0.030,  # in m
    'marker_length': 0.023,  # in m
    'num_px_height': 600,
    'num_px_width': 600,
    'aruco_dict': cv2.aruco.DICT_6X6_100
}

if __name__ == "__main__":

    # 0. LOADING RELEVANT IMAGES
    rgb_img = cv2.imread('data/rgb_images/20230725_143135_HoloLens.jpg')
    rgb_intrinsic, rgb_distort, rgb_extrinsic, rgb_pose, depth_intrinsic, depth_extrinsic, depth_pose = \
        load_hololens_matrices_at_time(time='2023-07-25_14-31-29')
    depth_map = get_depth_map(time='2023-07-25_14-31-29')
    # ----------

    # 1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
    board_specs = Board16x12
    board, aruco_params, aruco_dictionary = generate_charuco_board(specs=board_specs)
    # ----------

    # 2. DETECT CHARUCO BOARD CORNERS
    gray_img, overlay_img, aruco_corners, aruco_ids, charuco_corners, charuco_ids = detect_display_markers(
        charuco_board=board, img=rgb_img, aruco_dict=aruco_dictionary, aruco_param=aruco_params,
        cam_mtx=rgb_intrinsic, dist_coeffs=rgb_distort)

    # Draw detected corners on markers
    cv2.aruco.drawDetectedCornersCharuco(overlay_img, charuco_corners, charuco_ids, (0, 0, 255))
    plt.imshow(overlay_img)
    plt.title('Charuco corners and ids detected')
    plt.show()

    # Print the IDs and corner coordinates of the detected markers
    if aruco_ids is not None:
        centres = [np.mean(aruco_corners[i][0], axis=0) for i in range(len(aruco_ids))]
        [cv2.circle(gray_img, tuple(centre), 3, (0, 255, 0), -1) for centre in centres]
    # ----------

    # 3. OBTAIN POSE (RVEC AND TVEC) OF BOARD WITH RESPECT TO RGB CAMERA.
    outcome, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners, charucoIds=charuco_ids, board=board,
        cameraMatrix=rgb_intrinsic, distCoeffs=rgb_distort, rvec=None, tvec=None)

    # Get corresponding image points of rigid base corners from measured positions in board coordinates.
    T_bc = CharucoBoard_to_HoloLensRGB(rvec=rvecs, tvec=tvecs, pv_intrinsic=rgb_intrinsic)
    board_points_h = rigid_base_corners_on_board(board_spec=board_specs, cube_size=0.030)

    # Board to colour coordinates
    img_points, img_points_h = transform_from_X_to_Y(T_bc, board_points_h)

    # Display pose of board with respect to RGB camera
    [cv2.circle(rgb_img, tuple(img_points[idx, :].astype(int)), radius=5,
                color=(255, 255, 0), thickness=-1) for idx in range(img_points.shape[0])]
    cv2.drawFrameAxes(overlay_img, cameraMatrix=rgb_intrinsic, distCoeffs=rgb_distort,
                      rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
    cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Board pose", 960, 540)
    cv2.imshow("Board pose", overlay_img)
    cv2.waitKey(0)
    # -------------

    # 4. BOARD TO DEPTH SPACE

    # Collect all relevant transformations
    T_hc = rgb_extrinsic
    T_hd = depth_extrinsic
    T_bc = np.vstack((T_bc, [0., 0., 0., 1.]))
    T_cd = T_hd @ np.linalg.inv(T_hc)
    T_bd = T_hd @ np.linalg.inv(T_hc) @ T_bc

    # Board coordinates to depth coordinates
    corner_depth, corner_depth_homogenous = transform_from_X_to_Y(T_bd, board_points_h)

    # Colour to depth coordinates
    img_points_h_3d = homogeneous_coordinates(img_points_h, 'row')
    colour_depth, colour_depth_homogenous = transform_from_X_to_Y(T_cd, img_points_h_3d)

    # Depth to board coordinates
    corner_depth_board, _ = transform_from_X_to_Y(np.linalg.inv(T_bd), corner_depth_homogenous)
    colour_depth_board, _ = transform_from_X_to_Y(np.linalg.inv(T_bd), colour_depth_homogenous)
    # ------------

    # 5. TRANSFORM DEPTH DATA TO SHARED COORDINATE SYSTEM (BOARD COORDINATE SPACE)

    # Convert point cloud from HoloLens coordinate system to depth coordinates
    pcd = o3d.io.read_point_cloud('data/point_clouds/binary/depth_point-cloud_{}.ply'
                                  .format('2023-07-25_14-31-29'))
    world_pcd = np.asarray(pcd.points)
    world_pcd_h = homogeneous_coordinates(world_pcd, 'column').T
    world_to_depth, world_to_depth_h = transform_from_X_to_Y(T_hd, world_pcd_h)
    depth_est = world_to_depth[:, -1]

    # Depth to board coordinates
    depth_to_board, depth_to_board_h = transform_from_X_to_Y(np.linalg.inv(T_bd), world_to_depth_h)

    # Using depth map
    U, V = np.meshgrid(np.array([list(range(1, depth_map.shape[1] + 1))]),
                       np.array([list(range(depth_map.shape[0], 0, -1))]))
    depth_img = np.array([U.reshape(-1, 1), V.reshape(-1, 1), depth_map.reshape(-1, 1)]).T.reshape(-1, 3)
    depth_img_h = homogeneous_coordinates(depth_img, 'column').T
    depth_to_board, depth_to_board_h = transform_from_X_to_Y(np.linalg.inv(T_bd), depth_img_h)

    # Normalise colorbar
    color = 'magma'
    normalizer = mpl.colors.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap=color)

    # Compare depth to board transformed data vs original depth data
    fig = plt.figure()

    plt.contourf(U, V, depth_map, cmap=color, origin='upper')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.colorbar(mapper)
    plt.show()

    # ax_db = fig.add_subplot(1, 2, 1, projection='3d')
    # scatter_db = ax_db.scatter(depth_map_to_board[:, 0], depth_map_to_board[:, 1],
    #                            depth_map_to_board[:, 2], c=depth_map_to_board[:, 2], cmap='magma')
    # cbar_db = fig.colorbar(scatter_db, ax=ax_db)
    # cbar_db.set_label('depth estimate')
    # ax_db.set_xlabel('x')
    # ax_db.set_ylabel('y')
    # ax_db.set_zlabel('z')
    # ax_db.set_title('Depth map to board transformation')
    #
    # ax_d = fig.add_subplot(1, 2, 2, projection='3d')
    # scatter_d = ax_d.scatter(world_pcd[:, 0], world_pcd[:, 1],
    #                          world_pcd[:, 2], c=world_pcd[:, 2], cmap='magma')
    # cbar_d = fig.colorbar(scatter_d, ax=ax_d)
    # cbar_d.set_label('depth estimate')
    # ax_d.set_xlabel('x')
    # ax_d.set_ylabel('y')
    # ax_d.set_zlabel('z')
    # ax_d.set_title('Depth data')
    #
    # plt.show()
    # ---------

    # 6. COMPUTE EULER ANGLE FROM T_BD
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
