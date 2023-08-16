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
conf.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
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
        ir1_frame = f.get_infrared_frame(1)     # 1 is for left IR cam, 2 is for right IR cam.
        depth_frame = f.get_depth_frame()
        accel_frame = f.first_or_default(rs.stream.accel)
        gyro_frame = f.first_or_default(rs.stream.gyro)

        if ir1_frame:

            #   extract sensor intrinsics (https://github.com/IntelRealSense/librealsense/issues/10180)
            ir1_frame_intrinsic = ir1_frame.profile.as_video_stream_profile().get_intrinsics()
            ir1_intrinsic, ir1_distortion = construct_intrinsic(ir1_frame_intrinsic)

            #   extract transformation matrices from depth and RGB to accelerometer space
            ir1_to_accel_extrinsic = ir1_frame.profile.get_extrinsics_to(accel_frame.profile)
            accel_to_ir1_extrinsic = accel_frame.profile.get_extrinsics_to(ir1_frame.profile)
            T_accel_to_ir1 = construct_extrinsic(accel_to_ir1_extrinsic)

            #   collect image data if image frames are detected
            ir1_img = np.asanyarray(ir1_frame.get_data())

        if depth_frame:

            #   colourise depth cloud
            #   https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/pyglet_pointcloud_viewer.py#L409
            #   https://github.com/IntelRealSense/librealsense/issues/6194#issuecomment-608371293
            colorizer = rs.colorizer()
            colorized_depth = colorizer.colorize(depth_frame)
            depth_colormap = np.asanyarray(colorized_depth.get_data())

            #   export depth point cloud as PLY file (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/export_ply_example.py)


        else:
            continue

        #   collect imu data
        accel = accel_data(accel_frame.as_motion_frame().get_motion_data())
        gyro = gyro_data(gyro_frame.as_motion_frame().get_motion_data())

        #   show data
        cv2.imshow("Left IR", ir1_img)

        #   stop acquisition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#   save data
cv2.imwrite('data/realsense/ir1_img.png', ir1_img)
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
    charuco_board=board, img=ir1_img, aruco_dict=aruco_dictionary, aruco_param=aruco_params,
    cam_mtx=ir1_intrinsic, dist_coeffs=ir1_distortion)

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
    cameraMatrix=ir1_intrinsic, distCoeffs=ir1_distortion, rvec=None, tvec=None)

#   Get corresponding image points of rigid base corners from measured positions in board coordinates.
T_bc = ChArUco_pose_est.charucoboard_to_camspace(rvec=rvecs, tvec=tvecs, intrinsic=ir1_intrinsic)
rigid_board_points_h = ChArUco_pose_est.rigid_base_corners_on_board(board_spec=board_specs, cube_size=0.030, z=0.0)
rigid_board_points = rigid_board_points_h[:-1, :].T

#   Board to colour coordinates
img_points, img_points_h = ChArUco_pose_est.transform_from_X_to_Y(T_bc, rigid_board_points_h)

#   Display pose of board with respect to RGB camera
[cv2.circle(overlay_img, tuple(img_points[idx, :].astype(int)), radius=6,
            color=(255, 255, 0), thickness=-1) for idx in range(img_points.shape[0])]
cv2.drawFrameAxes(overlay_img, cameraMatrix=ir1_intrinsic, distCoeffs=ir1_distortion,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 960, 540)
cv2.imshow("Board pose", overlay_img)
cv2.waitKey(0)

#   Transform gravity vector from accel space to board space via depth space
T_accel_to_ir1 = np.vstack((T_accel_to_ir1, [0., 0., 0., 1.]))
accel_h = np.vstack(([accel, 0.]))
gravity_board = T_accel_to_ir1 @ accel_h
print(gravity_board)
