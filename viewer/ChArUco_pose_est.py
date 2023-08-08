import sys
import numpy as np
import pandas as pd
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
    # pv_pose = pd.read_csv('data/matrices/pv_pose_{}.csv'.format(time),
    #                       sep=',', header=None).values.T
    lt_intrinsic = pd.read_csv('data/matrices/LT_intrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T
    lt_extrinsic = pd.read_csv('data/matrices/LT_extrinsics_{}.csv'.format(time),
                               sep=',', header=None).values.T
    # lt_pose = pd.read_csv('data/matrices/lt_pose_{}.csv'.format(time),
    #                      sep=',', header=None).values.T
    lt_to_world = pd.read_csv('data/matrices/lt_to_world_{}.csv'.format(time),
                              sep=',', header=None).values.T
    world_to_pv = pd.read_csv('data/matrices/world_to_pv_{}.csv'.format(time),
                              sep=',', header=None).values.T

    return pv_intrinsic, pv_distort, pv_extrinsic, lt_intrinsic, lt_extrinsic, lt_to_world, world_to_pv


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


def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    # Alternative python implementation: https://www.appsloveworld.com/python/782/how-to-convert-a-rodrigues-vector-
    # to-a-rotation-matrix-without-opencv-using-pytho
    theta = np.linalg.norm(rodrigues_vec)

    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = np.array(rodrigues_vec / theta).flatten()
        I = np.eye(3, dtype=float)
        r_rT = r.reshape(-1, 1) @ r.reshape(-1, 3)

        r_cross = np.array([
            [0., -r[2], r[1]],
            [r[2], 0., -r[0]],
            [-r[1], r[0], 0.]
        ])

        # Formulation can be found in Liang (2018): "Efficient conversion from rotating matrix to
        # rotation axis and angle by extending Rodrigues' formula"
        rotation_mat = np.cos(theta) * I + (1 - np.cos(theta)) * r_rT + np.sin(theta) * r_cross

        r_cross_R = np.cos(theta) * r_cross + np.sin(theta) * r_cross @ r_cross
        zeta = np.arcsin(-np.trace(r_cross_R) / 2)

        angle_1 = theta
        angle_2 = zeta

    return rotation_mat, angle_1, angle_2


def CharucoBoard_to_HoloLensRGB(rvec, tvec, pv_intrinsic):
    """
    This function computes the transformation matrix from ChArUco board coordinates to the HoloLens RGB camera space.

    Parameters
    ----------
    rvec: array
        1x3 rotation vector showing rotation of object in Euler angles about the X, Y, Z axes with respect to the
        RGB camera.
    tvec: array
        1x3 translation vector from centre of RGB camera to object.
    pv_intrinsic: array
        4x4 calibration matrix of the RGB camera, containing the camera's focal length and principal point.

    Return
    ------
    t_bc: array
        4x4 transformation matrix from board coordinates to RGB camera space.
    """

    # Compute 3x3 rotation matrix from the Euler angles vector
    # Documentation for Rodrigues: https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
    rotation_mat, _ = cv2.Rodrigues(rvec)

    # Concatenate the translation vector with rotation matrix to compute the pose matrix
    pose_bc = np.hstack((rotation_mat, tvec))

    # Project the RGB camera calibration properties onto the pose matrix to obtain the
    # transformation from board to RGB spaces.
    t_bc = pv_intrinsic @ pose_bc

    return t_bc


def homogeneous_vectors(points, stack):
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


def rigid_base_corners_on_board(board_spec, cube_size, z):
    """
    This function computes the position of the rigid base corners of the cantilever beam in board coordinates.

    Parameters
    ---------
    board_spec: dictionary
        Contains the specifications to create a ChArUco board of m squares by n squares.

    cube_size: float
        Dimensions of the square cross-section of the cantilever beam attached to the rigid base.

    z: float
        Distance of rigid base corner from the surface viewed of the ChArUco board.

    Return
    ------
    homogeneous_vectors(board_points)
        4 x n array of rigid base coordinates in board space, where the last column is only 1's.
    """

    # Compute the centre point of the board, which is aligned with centre point of cantilever beam's fixed surface
    centre_pt = np.array([board_spec['squares_x'] * board_spec['square_length'],
                          board_spec['squares_y'] * board_spec['square_length']]) / 2

    # Obtain the position of the four corners of the cantilever beam's rigid base.
    board_points = np.array([[centre_pt[0] - cube_size / 2, centre_pt[1] - cube_size / 2, z],
                             [centre_pt[0] - cube_size / 2, centre_pt[1] + cube_size / 2, z],
                             [centre_pt[0] + cube_size / 2, centre_pt[1] + cube_size / 2, z],
                             [centre_pt[0] + cube_size / 2, centre_pt[1] - cube_size / 2, z]])

    return homogeneous_vectors(board_points, 'column').T


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


def matrix_minus_vector(matrix, vector):
    """
    This function subtracts a vector from a 2D matrix.

    Parameters
    ----------
    matrix: array
        3 x n array containing coordinates of n nodes on an object
    vector: array
        3 x 1 array containing the centroid position of said object

    Returns
    -------
    new_matrix: array
        3 x n array of n centred points on an object
    """

    new_matrix = np.array([matrix[idx] - vector[idx] for idx in range(len(matrix))])
    return new_matrix


def find_centroid(matrix):
    """
    This function computes the centroid position of an object, given it's 3D nodal coordinates.

    Parameters
    ----------
    matrix: array
        3 x n array containing coordinates of n nodes on an object

    Return
    -----
    centroid: array
        3 x 1 array containing the centroid position of said object

    """

    centroid = np.array([np.mean(matrix[idx]) for idx in range(len(matrix))])
    return centroid


def find_transformation(experimental_corners, model_corners):
    """
    Finds the transformation (rotation and translation) between the experimental and model back corner points,
    specifically transforming model_corners into experimental corners.

    Parameters
    ----------
    experimental_corners: array
        orientation data from experimental points
    model_corners: array
        orientation data from model

    Returns
    -------
    R: array
       3 x 3 rotation matrix between model and experimental data
    t: array
       3 x 1 translation vector between the origins of model space and experimental data space
    """

    # Organise points into [x1,x2,x3; y1,y2,y3; z1,z2,z3] format
    model_corners = model_corners.transpose()
    experimental_corners = experimental_corners.transpose()

    # Find centroids of points
    model_centroid = find_centroid(model_corners)
    experimental_centroid = find_centroid(experimental_corners)

    # Remove translation
    model_centered = matrix_minus_vector(model_corners, model_centroid)
    experimental_centered = matrix_minus_vector(experimental_corners, experimental_centroid)

    # Calculate rotation matrix using Horns quarterion with horns least square from Horn, 1986
    # source: https://pdfs.semanticscholar.org/3120/a0e44d325c477397afcf94ea7f285a29684a.pdf
    M = model_centered @ experimental_centered.transpose()

    [Sxx, Sxy, Sxz] = M[0]
    [Syx, Syy, Syz] = M[1]
    [Szx, Szy, Szz] = M[2]

    # Create N matrix
    N = np.vstack(([(Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx)],
                   [(Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz)],
                   [(Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy)],
                   [(Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz)]))

    # Find the maximum eigenvector which is the rotation quarterion
    [evalues, evectors] = np.linalg.eig(N)
    emax = np.max(np.abs(evalues))
    i_emax = np.unravel_index(np.argmax(np.real(evalues)), evalues.shape)  # index of largest eigenvector
    q = np.real(evectors[:, i_emax[0]])
    sign = np.sign(np.max(np.abs(q)))  # sign ambiguity
    q = q * sign

    # Find orthogonal rotation matrix
    unit_q = q / np.linalg.norm(q)  # unit quarterion
    [q0, qx, qy, qz] = unit_q
    v = np.array((qx, qy, qz))
    Z = np.vstack(([q0, -qz, qy],
                   [qz, q0, -qx],
                   [-qy, qx, q0]))

    R = v.reshape(3, 1) @ v.reshape(1, 3) + np.matmul(Z, Z)  # reshape vector for matrix multiplication

    # Find translation from centroids
    t = experimental_centroid - R @ model_centroid

    return R, t


def align_model(R, t, model_points):
    """
    Aligns the original model points to the experimental data (NOTE: it performs the same functionality
    as transform_from_X_to_Y).

    Parameters
    ---------
    R: array
        3 x 3 rotation matrix between model and experimental data
    t: array
        3 x 1 translation vector between the origins of model space and experimental data space
    model_points: array
        original model points in coordinate format

    Return
    ------
    transformed_points.transpose(): array
        transformed points in coordinate format
    """
    # model points transposed to use matmul transformation
    model_points = model_points.transpose()

    rotated_points = np.matmul(R, model_points)
    transformed_points = matrix_minus_vector(rotated_points, -t)

    # transformed points transposed to return to coordinate format
    return transformed_points.transpose()


def reorder_model_node_coordinates(model_node_coordinates, corner):
    """
    This function changes the order of the model node coordinates in the horizontal axis (i.e. columns) from [z, x, y]
    to [x, y, z]. Note that x is the width of cross-section, y is the height of cross-section, and  z is the length
    along beam.

    Parameters
    ---------
    model_node_coordinates: array
        n x 3 array for nodal coordinates of model in the order [z, x, y].

    corner: boolean
        specifies if model_node_coordinates are corner points or not.

    Return
    -----
    reordered_model_nodes: array
        n x 3 array for nodal coordinates of model in the order [x, y, z].

    """

    if corner is True:
        # Order node coordinates to go clockwise from top left corner of cross-section,
        # and convert measurements from mm to m.
        reordered_model_nodes = np.array([model_node_coordinates[1], model_node_coordinates[0],
                                          model_node_coordinates[2], model_node_coordinates[3]]) / 1000
    else:
        reordered_model_nodes = model_node_coordinates / 1000

    # Swap columns around to order [x, y, z]
    reordered_model_nodes = np.array([reordered_model_nodes[:, 1], reordered_model_nodes[:, 2],
                                      reordered_model_nodes[:, 0]]).T

    return reordered_model_nodes


def create_point_cloud(filepath, points):
    """
    This function creates a 3D point cloud from a 2D array of points, and writes the point cloud
    to a .ply file specified in the filepath.

    Parameters
    ---------
    filepath: str
        path to where the .ply point cloud file is saved.

    points: array
        k x 3 array of nodal coordinates of points in point cloud.
    """

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pc)


# ChArUco board specs (m squares x n squares)
Board16x12 = {
    'squares_x': 16,  # number of squares (n) in the x direction (rows)
    'squares_y': 12,  # number of squares (m) in the y direction (columns)
    'square_length': 0.016,  # length of chessboard squares in m
    'marker_length': 0.012,  # length of ArUco marker in m
    'num_px_height': 600,  # number of pixels in the x direction
    'num_px_width': 600,  # number of pixels in the y direction
    'aruco_dict': cv2.aruco.DICT_6X6_100  # ArUco dictionary with 100 different unique marjers, each marker in
    # dictionary has a distinct pattern. This pattern is defined by a
    # grid of 6x6 black and white cells
}
Board12x9 = {
    'squares_x': 12,
    'squares_y': 9,
    'square_length': 0.022,
    'marker_length': 0.017,
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

    # 0. LOADING RELEVANT IMAGES AND CAMERA MATRICES

    # time at which HoloLens Research Mode images were taken
    imtime = '2023-07-25_14-31-29'

    # RGB images are captured slightly after the research mode images
    rgb_img = cv2.imread('data/rgb_images/20230725_143135_HoloLens.jpg')

    rgb_intrinsic, rgb_distort, rgb_extrinsic, depth_intrinsic, depth_extrinsic, T_dw, T_wc = \
        load_hololens_matrices_at_time(time=imtime)

    depth_map = get_depth_map(time=imtime)
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

    R, theta, zeta = rodrigues_vec_to_rotation_mat(rodrigues_vec=rvecs)
    gravity = 9.81 * np.array([np.sin(theta), np.cos(theta) * np.cos(0), np.cos(theta) * np.sin(0)]) # note that we swapped x and y around

    # Get corresponding image points of rigid base corners from measured positions in board coordinates.
    T_bc = CharucoBoard_to_HoloLensRGB(rvec=rvecs, tvec=tvecs, pv_intrinsic=rgb_intrinsic)
    rigid_board_points_h = rigid_base_corners_on_board(board_spec=board_specs, cube_size=0.030, z=0.0)
    rigid_board_points = rigid_board_points_h[:-1, :].T

    # Board to colour coordinates
    img_points, img_points_h = transform_from_X_to_Y(T_bc, rigid_board_points_h)

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
    corner_depth, corner_depth_homogenous = transform_from_X_to_Y(T_bd, rigid_board_points_h)

    # Colour to depth coordinates
    img_points_h_3d = homogeneous_vectors(img_points_h, 'row')
    colour_depth, colour_depth_homogenous = transform_from_X_to_Y(T_cd, img_points_h_3d)

    # Depth to board coordinates
    corner_depth_board, _ = transform_from_X_to_Y(np.linalg.inv(T_bd), corner_depth_homogenous)
    colour_depth_board, _ = transform_from_X_to_Y(np.linalg.inv(T_bd), colour_depth_homogenous)
    # ------------

    # 5. TRANSFORM BOARD (B), RGB CAM (C), AND DEPTH CAM DATA ONTO REAL WORLD SPACE

    # Collect all transformations after considering world coordinates
    T_cw = np.linalg.inv(T_wc)
    T_bw = T_cw @ T_bc
    T_cd = np.linalg.inv(T_dw) @ T_cw
    T_dc = np.linalg.inv(T_cd)
    T_bd = T_cd @ T_bc

    # Align rigid base corners in board space to corresponding positions in depth space.
    b_in_d, _ = transform_from_X_to_Y(T_bd, rigid_board_points_h)

    # 6. VISUALISE RIGID PLATE CORNERS WITH POINT CLOUD

    # Point cloud 1: backplate corners on board in depth space
    create_point_cloud("data/point_clouds/binary/backplate_corners_depth.ply", b_in_d)

    # 7. ALIGN CANTILEVER GEOMETRY TO DEPTH POINT CLOUD SPACE

    # Load in all model information
    model_nodes = pd.read_csv(
        'data/cantilever/model_nodes.csv', sep=',', header=None).values
    model_corner_nodes = pd.read_csv(
        'data/cantilever/model_fixed_end_corners.csv', sep=',', header=None).values
    transformed_model_nodes = pd.read_csv(
        'data/cantilever/transformed_model_nodes.csv', sep=',', header=None).values
    transformed_model_corner_nodes = pd.read_csv(
        'data/cantilever/transformed_model_fixed_end_corners.csv', sep=',', header=None).values
    centred_model_nodes = pd.read_csv(
        'data/cantilever/centred_transformed_model_nodes.csv', sep=',', header=None).values
    deformed_model = pd.read_csv(
        'data/cantilever/deformed_model_nodes.csv', sep=',', header=None).values

    # a. Toy example: model fixed end corners
    # Process corner nodes for subsequent transformation between spaces
    model_corner_nodes = reorder_model_node_coordinates(model_corner_nodes, True)

    # Compute transformation between board and model spaces
    rotation, translation = find_transformation(rigid_board_points, model_corner_nodes)
    T_mb = np.vstack((np.hstack((rotation, translation.reshape(-1, 1))), [0., 0., 0., 1.]))

    # Infer transformation between model and depth spaces
    T_md = T_bd @ T_mb
    corners_in_depth, corners_in_depth_h = transform_from_X_to_Y(
        T_md, homogeneous_vectors(model_corner_nodes, 'column').T)

    # Point cloud 2: model fixed end in depth space.
    create_point_cloud("data/point_clouds/binary/backplate_corners_from_model.ply",
                       corners_in_depth)

    # b. Transform all model nodes of the undeformed geometry to depth space
    undeformed_model_nodes = reorder_model_node_coordinates(model_nodes, False)
    undeformed_model_in_depth, _ = transform_from_X_to_Y(
        T_md, homogeneous_vectors(undeformed_model_nodes, 'column').T)

    # Point cloud 3: undeformed model in depth space
    create_point_cloud("data/point_clouds/binary/model_points_in_depth.ply",
                       undeformed_model_in_depth)

    # c. Transform all model nodes of the deformed geometry to depth space
    # Transformation matrix between deformed geometry to undeformed geometry by aligning the rigid base corners
    # of the deformed geometry to the undeformed geometry (in depth space) using MeshLab's align tool.
    T_deformed_undeformed = np.array([[0., 0.33, -0.94, -0.02],
                                      [-0.97, -0.22, -0.08, -0.18],
                                      [-0.24, 0.92, 0.32, 0.52],
                                      [0., 0., 0., 1.]])
    deformed_model_nodes = reorder_model_node_coordinates(deformed_model, False)
    deformed_model_in_depth, deformed_model_in_depth_h = transform_from_X_to_Y(
        T_deformed_undeformed, homogeneous_vectors(deformed_model_nodes, 'column').T)

    # Point cloud 4: deformed model geometry in depth space
    create_point_cloud("data/point_clouds/binary/deformed_model_in_depth.ply",
                       deformed_model_in_depth)


    # 8. CONVERT DEPTH INFORMATION TO UNDEFORMED MODEL SPACE
    T_dm = np.linalg.inv(T_md)

    # Depth point cloud to undeformed space
    pc = o3d.io.read_point_cloud('data/point_clouds/binary/depth_point-cloud_{}.ply'.format(imtime))
    depth_pc = np.asarray(pc.points)
    depth_pc_h = homogeneous_vectors(depth_pc, 'column').T

    d_in_m, _ = transform_from_X_to_Y(T_dm, depth_pc_h)
    corners_in_m, _ = transform_from_X_to_Y(T_dm, corners_in_depth_h)
    deformed_in_m, _ = transform_from_X_to_Y(T_dm, deformed_model_in_depth_h)

    create_point_cloud("data/point_clouds/binary/cantilever_model.ply", undeformed_model_nodes*1000)
    create_point_cloud("data/point_clouds/binary/TEST_depth_points_in_model.ply", d_in_m*1000)
    create_point_cloud("data/point_clouds/binary/TEST_rigid_corner_in_model.ply", corners_in_m*1000)
    create_point_cloud("data/point_clouds/binary/TEST_deformed_geometry_in_model.ply", deformed_in_m*1000)
