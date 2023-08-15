import numpy as np
import pyrealsense2 as rs
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
conf.enable_stream(rs.stream.accel)
conf.enable_stream(rs.stream.gyro)

#   start streaming
pipeline.start(conf)

try:
    while True:
        #   wait for a frame (make sure camera not opened on RealSense viewer)
        f = pipeline.wait_for_frames()

        #   collect image frames and IMU data frames
        RGB_frame = f.get_color_frame()
        depth_frame = f.get_depth_frame()
        accel_frame = f.first_or_default(rs.stream.accel)
        gyro_frame = f.first_or_default(rs.stream.gyro)

        if RGB_frame and depth_frame:

            #   extract sensor intrinsics (https://github.com/IntelRealSense/librealsense/issues/10180)
            depth_intrinsic = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            RGB_intrinsic = RGB_frame.profile.as_video_stream_profile().get_intrinsics()
            # RGB_distortion =
            depth_to_RGB_extrinsic = depth_frame.profile.get_extrinsics_to(RGB_frame.profile)

            #   collect image data if image frames are detected
            RGB_img = np.asanyarray(RGB_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            #   improve appearance of rgb image
            RGB_norm = cv2.normalize(
                RGB_img, np.zeros((RGB_img.shape[0], RGB_img.shape[1])), 0, 255, cv2.NORM_MINMAX)

            #   create depth map for visualisation
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

            #   obtain scale of depth values in depth frame
            depth_sensor = pipeline.start().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            #   compute depth distance for any given point (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py)


            #   export depth point cloud as PLY file (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/export_ply_example.py)


        else:
            continue

        #   collect imu data
        accel = accel_data(accel_frame.as_motion_frame().get_motion_data())
        gyro = gyro_data(gyro_frame.as_motion_frame().get_motion_data())

        #   show data
        cv2.imshow("RGB", RGB_img)
        cv2.imshow("RGB norm", RGB_norm)
        cv2.imshow("Depth norm", depth_colormap)

        #   stop acquisition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#   save data
cv2.imwrite('data/realsense/rgb_img.png', RGB_img)
#cv2.imwrite('data/realsense/depth_norm.png', depth_colormap)
#ChArUco_pose_est.write_to_csv('data/realsense/depth_norm.csv', depth_colormap)
ChArUco_pose_est.write_to_csv('data/realsense/gravity.csv', accel)

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


# 1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
board_specs = Board16x12
board, aruco_params, aruco_dictionary = ChArUco_pose_est.generate_charuco_board(specs=board_specs)
# ----------

# 2. DETECT CHARUCO BOARD CORNERS
gray_img, overlay_img, aruco_corners, aruco_ids, charuco_corners, charuco_ids = ChArUco_pose_est.detect_display_markers(
    charuco_board=board, img=RGB_img, aruco_dict=aruco_dictionary, aruco_param=aruco_params,
    cam_mtx=RGB_intrinsic, dist_coeffs=RGB_distort)
