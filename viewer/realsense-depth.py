import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import ChArUco_pose_est


#   NOTE: MOVE THIS FILE AND ALL REALSENSE DATA TO NEW REPO, MAKE SURE TO PRESERVE HISTORY
#   (https://medium.com/@ayushya/move-directory-from-one-repository-to-another-preserving-git-history-d210fa049d4b)

#   functions

#   returns camera's angular velocity (rad/s).
def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


#   returns linear acceleration of camera (m/s^2).
def accel_data(accel):
    return np.asarray([[accel.x],
                       [accel.y],
                       [accel.z]])


def construct_intrinsic(intrinsic_frame):
    """

    :param intrinsic_frame:
    :return:
    """

    intrin_mtx = np.array([[intrinsic_frame.fx, 0., intrinsic_frame.ppx],
                           [0., intrinsic_frame.fy, intrinsic_frame.ppy],
                           [0., 0., 1.]])

    distort_mtx = np.array(intrinsic_frame.coeffs)

    return intrin_mtx, distort_mtx


def construct_extrinsic(extrinsic_frame):
    """

    :param extrinsic_frame:
    :return:
    """

    R = np.array([[extrinsic_frame.rotation[0:3]],
                  [extrinsic_frame.rotation[3:6]],
                  [extrinsic_frame.rotation[6:]]]).reshape(-1, 3)

    t = np.array([[extrinsic_frame.translation[0]],
                  [extrinsic_frame.translation[1]],
                  [extrinsic_frame.translation[2]]])

    T_mtx = np.hstack((R, t))

    return T_mtx


def write_to_csv(filepath, data):
    """
    This function writes a 2D array of data points to a CSV file located at a specified filepath.

    Parameters
    ----------
    filepath: str
        path to where the .csv data file is saved.

    data: array
        k x 3 array of nodal coordinates of data points.
    """

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.asarray(data))


#   initialise pipeline
pipeline = rs.pipeline()
conf = rs.config()
conf.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
conf.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
# conf.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
conf.enable_stream(rs.stream.accel)
conf.enable_stream(rs.stream.gyro)

#   start streaming
pipeline.start(conf)


#   REPLACE WITH IR STREAM INSTEAD OF RGB AND DEPTH

try:
    while True:
        #   wait for a frame (make sure camera not opened on RealSense viewer)
        f = pipeline.wait_for_frames()

        #   collect image frames and IMU data frames
        RGB_frame = f.get_color_frame()
        depth_frame = f.get_depth_frame()
        # ir1_frame = f.get_infrared_frame(1)
        accel_frame = f.first_or_default(rs.stream.accel)
        gyro_frame = f.first_or_default(rs.stream.gyro)

        if RGB_frame and depth_frame:

            #   extract sensor intrinsics (https://github.com/IntelRealSense/librealsense/issues/10180)
            depth_frame_intrinsic = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            RGB_frame_intrinsic = RGB_frame.profile.as_video_stream_profile().get_intrinsics()
            depth_intrinsic, depth_distortion = construct_intrinsic(depth_frame_intrinsic)
            RGB_intrinsic, RGB_distortion = construct_intrinsic(RGB_frame_intrinsic)

            #   extract transformation matrices from depth and RGB to accelerometer space
            depth_to_accel_extrinsic = depth_frame.profile.get_extrinsics_to(accel_frame.profile)
            RGB_to_accel_extrinsic = RGB_frame.profile.get_extrinsics_to(accel_frame.profile)
            accel_to_RGB_extrinsic = accel_frame.profile.get_extrinsics_to(RGB_frame.profile)
            accel_to_depth_extrinsic = accel_frame.profile.get_extrinsics_to(depth_frame.profile)
            T_depth_to_accel = construct_extrinsic(depth_to_accel_extrinsic)
            T_RGB_to_accel = construct_extrinsic(RGB_to_accel_extrinsic)
            accel_to_RGB = construct_extrinsic(accel_to_RGB_extrinsic)
            accel_to_depth = construct_extrinsic(accel_to_depth_extrinsic)

            #   collect image data if image frames are detected
            RGB_img = np.asanyarray(RGB_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            # ir1_img = np.asanyarray(ir1_frame.get_data())

            #   improve appearance of rgb image
            # RGB_norm = cv2.normalize(
            #     RGB_img, np.zeros((RGB_img.shape[0], RGB_img.shape[1])), 0, 255, cv2.NORM_MINMAX)
            #
            # #   create depth map for visualisation
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

            #   obtain scale of depth values in depth frame
            # depth_sensor = f.get_device().first_depth_sensor()
            # depth_scale = depth_sensor.get_depth_scale()

            #   compute depth distance for any given point (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py)

            #   export depth point cloud as PLY file (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/export_ply_example.py)


        else:
            continue

        #   collect imu data
        accel = accel_data(accel_frame.as_motion_frame().get_motion_data())
        gyro = gyro_data(gyro_frame.as_motion_frame().get_motion_data())

        #   show data
        cv2.imshow("RGB", RGB_frame)
        # cv2.imshow("Left IR", ir1_img)

        #   stop acquisition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#   save data
# cv2.imwrite('data/realsense/rgb_img.png', RGB_img)
# cv2.imwrite('data/realsense/depth_norm.png', depth_colormap)
# ChArUco_pose_est.write_to_csv('data/realsense/depth_norm.csv', depth_colormap)
# ChArUco_pose_est.write_to_csv('data/realsense/gravity.csv', accel)

#   ChArUco board specs (m squares x n squares)
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

#   1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
board_specs = Board16x12
board, aruco_params, aruco_dictionary = ChArUco_pose_est.generate_charuco_board(specs=board_specs)
# ----------

#   2. DETECT CHARUCO BOARD CORNERS
gray_img, overlay_img, aruco_corners, aruco_ids, charuco_corners, charuco_ids = ChArUco_pose_est.detect_display_markers(
    charuco_board=board, img=RGB_img, aruco_dict=aruco_dictionary, aruco_param=aruco_params,
    cam_mtx=RGB_intrinsic, dist_coeffs=RGB_distortion)

#   Draw detected corners on markers
cv2.aruco.drawDetectedCornersCharuco(overlay_img, charuco_corners, charuco_ids, (0, 0, 255))
plt.imshow(overlay_img)
plt.title('Charuco corners and ids detected')
plt.show()

#   Print the IDs and corner coordinates of the detected markers
if aruco_ids is not None:
    centres = [np.mean(aruco_corners[i][0], axis=0) for i in range(len(aruco_ids))]
    [cv2.circle(gray_img, tuple(centre), 3, (0, 255, 0), -1) for centre in centres]

#   3. OBTAIN POSE (RVEC AND TVEC) OF BOARD WITH RESPECT TO DEPTH SENSOR.
outcome, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
    charucoCorners=charuco_corners, charucoIds=charuco_ids, board=board,
    cameraMatrix=RGB_intrinsic, distCoeffs=RGB_distortion, rvec=None, tvec=None)

#   Get corresponding image points of rigid base corners from measured positions in board coordinates.
T_bc = ChArUco_pose_est.charucoboard_to_camspace(rvec=rvecs, tvec=tvecs, intrinsic=RGB_intrinsic)
rigid_board_points_h = ChArUco_pose_est.rigid_base_corners_on_board(board_spec=board_specs, cube_size=0.030, z=0.0)
rigid_board_points = rigid_board_points_h[:-1, :].T

#   Board to colour coordinates
img_points, img_points_h = ChArUco_pose_est.transform_from_X_to_Y(T_bc, rigid_board_points_h)

#   Display pose of board with respect to RGB camera
[cv2.circle(overlay_img, tuple(img_points[idx, :].astype(int)), radius=6,
            color=(255, 255, 0), thickness=-1) for idx in range(img_points.shape[0])]
cv2.drawFrameAxes(overlay_img, cameraMatrix=RGB_intrinsic, distCoeffs=RGB_distortion,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 960, 540)
cv2.imshow("Board pose", overlay_img)
cv2.waitKey(0)

#   Transform gravity vector from accel space to board space via depth space
accel_to_RGB = np.vstack((accel_to_RGB, [0., 0., 0., 0.]))
accel_h = np.vstack(([accel, 0.]))
accel_in_RGB = accel_to_RGB @ accel_h
print(accel_in_RGB)

T_bc = np.vstack((T_bc, [0., 0., 0., 1.]))
T_cb = np.linalg.inv(T_bc[:-1, :-1])
print(T_cb)

gravity_board = T_cb.T @ accel_in_RGB[:-1]
print(gravity_board)
