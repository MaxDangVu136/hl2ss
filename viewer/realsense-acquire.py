import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import open3d as o3d
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

    distort_mtx = np.array(intrinsic_frame.coeffs).reshape(-1, 1)

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


#   initialise pipeline
pipeline = rs.pipeline()
conf = rs.config()

depth = False
rgb_ir_imu = True

# conf.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
conf.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
conf.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
conf.enable_stream(rs.stream.accel)

align = rs.align(rs.stream.color)

#   start streaming
pipeline.start(conf)

try:
    while True:
        #   wait for a frame (make sure camera not opened on RealSense viewer)
        f = pipeline.wait_for_frames()

        if depth:
            #  collect data frames
            depth_frame = f.get_depth_frame()
            color_frame = f.get_color_frame()
            accel_frame = f.first_or_default(rs.stream.accel)
            aligned_frames = align.process(f)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()

            #   extract sensor intrinsics (https://github.com/IntelRealSense/librealsense/issues/10180)
            depth_frame_intrinsic = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            depth_intrinsic, depth_distortion = construct_intrinsic(depth_frame_intrinsic)

            #   extract transformation matrices from depth and RGB to accelerometer space
            # depth_to_RGB_extrinsic = depth_frame.profile.get_extrinsics_to(RGB_frame.profile)
            # T_depth_to_RGB = construct_extrinsic(depth_to_RGB_extrinsic)
            depth_to_accel_extrinsic = depth_frame.profile.get_extrinsics_to(accel_frame.profile)
            T_depth_to_accel = construct_extrinsic(depth_to_accel_extrinsic)

            aligned_color_to_depth = aligned_color_frame.profile.get_extrinsics_to(depth_frame.profile)
            T_aligned_color_depth = construct_extrinsic(aligned_color_to_depth)

            aligned_depth_to_color = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
            T_aligned_depth_color = construct_extrinsic(aligned_depth_to_color)

            aligned_color_to_accel = aligned_color_frame.profile.get_extrinsics_to(accel_frame.profile)
            T_aligned_color_accel = construct_extrinsic(aligned_color_to_accel)

            aligned_depth_aligned_color = aligned_depth_frame.profile.get_extrinsics_to(aligned_color_frame.profile)
            T_aligned_dc = construct_extrinsic(aligned_depth_aligned_color)

            # depth_to_ir1_extrinsic = depth_frame.profile.get_extrinsics_to(ir1_frame.profile)
            # T_depth_to_ir1 = construct_extrinsic(depth_to_ir1_extrinsic)

            # #   collect image data if image frames are detected
            # depth_img = np.asanyarray(depth_frame.get_data())

            #   display aligned rgb image and change color mapping
            img = np.asarray(aligned_color_frame.get_data())
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Video", cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("Video", img)

        elif rgb_ir_imu:
            RGB_frame = f.get_color_frame()
            ir1_frame = f.get_infrared_frame(1)     # 1 is for left IR cam, 2 is for right IR cam.
            accel_frame = f.first_or_default(rs.stream.accel)

            #   extract sensor intrinsics (https://github.com/IntelRealSense/librealsense/issues/10180)
            RGB_frame_intrinsic = RGB_frame.profile.as_video_stream_profile().get_intrinsics()
            RGB_intrinsic, RGB_distortion = construct_intrinsic(RGB_frame_intrinsic)
            ir1_frame_intrinsic = ir1_frame.profile.as_video_stream_profile().get_intrinsics()
            ir1_intrinsic, ir1_distortion = construct_intrinsic(ir1_frame_intrinsic)

            #   extract transformation matrices from depth and RGB to accelerometer space
            accel_to_ir1_extrinsic = accel_frame.profile.get_extrinsics_to(ir1_frame.profile)
            T_accel_to_ir1 = construct_extrinsic(accel_to_ir1_extrinsic)
            ir1_to_RGB_extrinsic = ir1_frame.profile.get_extrinsics_to(RGB_frame.profile)
            T_ir1_to_RGB = construct_extrinsic(ir1_to_RGB_extrinsic)

            #   collect image data if image frames are detected
            RGB_img = np.asanyarray(RGB_frame.get_data())
            ir1_img = np.asanyarray(ir1_frame.get_data())
            accel = accel_data(accel_frame.as_motion_frame().get_motion_data())

            #   show data
            cv2.imshow("Left IR", ir1_img)
            cv2.imshow("RGB image", RGB_img)

        else:
            continue

        #   stop acquisition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#%%
if depth:
    #   save acceleration to depth transformations
    ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_depth_accel.csv", T_depth_to_accel)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_aligned_color_accel.csv", T_aligned_color_accel)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_aligned_depth_accel.csv", T_aligned_depth_accel)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_depth_ir.csv")
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_depth_RGB.csv", T_depth_to_RGB)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_aligned_dc.csv", T_aligned_dc)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_aligned_color_depth.csv", T_aligned_color_depth)
    # ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_aligned_depth_color.csv", T_aligned_depth_color)

    #   declare pointcloud object, for calculating pointclouds and texture mappings
    pc = rs.pointcloud()

    #   mapping texture onto point cloud
    pc.map_to(aligned_color_frame)

    #   extract point coordinates from depth cloud
    points = pc.calculate(aligned_depth_frame)
    texture = np.asarray(aligned_color_frame.get_data())
    points.export_to_ply("data/realsense/point_cloud_depth.ply", aligned_color_frame)
    # texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)

    #   measuring distance of points from depth data
    #   https://github.com/IntelRealSense/librealsense/blob/master/examples/C/distance/rs-distance.c
    #   https://dev.intelrealsense.com/docs/rs-measure
    #   Note: All stereo-based 3D cameras have the property of noise being proportional to distance squared.
    # To counteract this we transform the frame into disparity-domain making the noise more uniform across distance

if rgb_ir_imu:
    #   save image data
    cv2.imwrite('data/realsense/image_data/ir1_img.png', ir1_img)
    cv2.imwrite('data/realsense/image_data/RGB_img.png', RGB_img)
    ChArUco_pose_est.write_to_csv('data/realsense/image_data/ir1_img.csv', ir1_img)
    ChArUco_pose_est.write_to_csv('data/realsense/image_data/RGB_img.csv', RGB_img)

#   ChArUco board specs (m squares x n squares)
Board16x12 = {
    'squares_x': 16,  # number of squares (n) in the x direction (rows)
    'squares_y': 12,  # number of squares (m) in the y direction (columns)
    'square_length': 0.016,  # length of chessboard squares in m
    'marker_length': 0.012,  # length of ArUco marker in m
    'num_px_height': 600,  # number of pixels in the x direction
    'num_px_width': 600,  # number of pixels in the y direction
    'aruco_dict': cv2.aruco.DICT_6X6_100  # ArUco dictionary with 100 different unique markers, each marker in
    # dictionary has a distinct pattern. This pattern is defined by a
    # grid of 6x6 black and white cells
}

#   CONVERT THESE TO FUNCTIONS

#   1. GENERATE CHARUCO BOARD AND THEIR OBJECT POINTS (IN BOARD COORDINATE SYSTEM)
board_specs = Board16x12
board, aruco_params, aruco_dictionary = ChArUco_pose_est.generate_charuco_board(specs=board_specs)

# #   Generate Charuco board image
# img_size = (800, 600)
# charuco_img = board.draw(img_size)
#
# #   Display the Charuco board
# plt.imshow(charuco_img, cmap='gray')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('ChArUco board')
# plt.show()
# # ----------

#%%
#   2. DETECT CHARUCO BOARD CORNERS
gray_img, overlay_img, aruco_corners, aruco_ids, charuco_corners, charuco_ids = \
    ChArUco_pose_est.detect_display_markers(charuco_board=board, img=RGB_img, aruco_dict=aruco_dictionary,
                                            aruco_param=aruco_params, cam_mtx=RGB_intrinsic,
                                            dist_coeffs=RGB_distortion)

#   Draw detected corners on markers
cv2.aruco.drawDetectedCornersCharuco(overlay_img, charuco_corners, charuco_ids, (0, 0, 255))
plt.imshow(overlay_img, cmap='gray')
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

#%%
#   Get corresponding image points of rigid base corners from measured positions in board coordinates.
T_board_to_RGB = ChArUco_pose_est.charucoboard_to_camspace(rvec=rvecs, tvec=tvecs, intrinsic=RGB_intrinsic)
ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_board_RGB.csv", T_board_to_RGB)
rigid_board_points_h = ChArUco_pose_est.rigid_base_corners_on_board(board_spec=board_specs, cube_size=0.030, z=0.0)
rigid_board_points = rigid_board_points_h[:-1, :].T
ChArUco_pose_est.write_to_csv("data/realsense/simulation_data/rigid_corners_board.csv", rigid_board_points_h.T)

#   Board to colour coordinates
img_points, img_points_h = ChArUco_pose_est.transform_from_X_to_Y(T_board_to_RGB, rigid_board_points_h)

#   Display pose of board with respect to RGB camera
[cv2.circle(overlay_img, tuple(img_points[idx, :].astype(int)), radius=10,
            color=(255, 255, 0), thickness=-1) for idx in range(img_points.shape[0])]
cv2.drawFrameAxes(overlay_img, cameraMatrix=RGB_intrinsic, distCoeffs=RGB_distortion,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.namedWindow("Board pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board pose", 960, 540)
cv2.imshow("Board pose", overlay_img)
cv2.waitKey(0)

# #   Transform gravity vector from accel space to board space via infrared camera space
T_board_to_RGB = np.vstack((T_board_to_RGB, [0., 0., 0., 1.]))
ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_board_RGB.csv", T_board_to_RGB)

T_accel_to_ir1 = np.vstack((T_accel_to_ir1, [0., 0., 0., 1.]))
ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_accel_ir1.csv", T_accel_to_ir1)

T_ir1_to_RGB = np.vstack((RGB_intrinsic @ T_ir1_to_RGB, [0., 0., 0., 1.]))
T_accel_to_RGB = T_ir1_to_RGB @ T_accel_to_ir1

T_ir1_to_board = np.linalg.inv(T_board_to_RGB) @ T_ir1_to_RGB
T_accel_to_board = T_ir1_to_board @ T_accel_to_ir1

accel_h = np.vstack(([accel, 0.]))
print("gravity IMU:", accel)
ChArUco_pose_est.write_to_csv("data/realsense/transformations/T_accel_board.csv", T_accel_to_board)
ChArUco_pose_est.write_to_csv("data/realsense/transformations/gravity.csv", accel.T)

#   x,y,z is right hand coordinate system, so flip y and z axis directions to match board orientation.
gravity_accel_board = T_accel_to_board @ accel_h
print("gravity in board:", gravity_accel_board)
gravity_unit = gravity_accel_board/np.linalg.norm(gravity_accel_board)
print("unit gravity:", gravity_unit)
gravity_info = np.array([gravity_accel_board[0], -gravity_accel_board[1], -gravity_accel_board[2]])
print("gravity info:", gravity_info.T)

#   Visualise gravity vector in RGB coordinates, inspect if it makes sense,
gravity_in_RGB, _ = ChArUco_pose_est.transform_from_X_to_Y(T_board_to_RGB[:-1, :], gravity_unit)
gravity_in_RGB = np.reshape(gravity_in_RGB, (-1, 1))
print("gravity in RGB:", gravity_in_RGB)

#   Calculate end point of vector to draw on image.
scale = 0.1
thickness = 5
color = (0, 165, 255) # BGR color (red)
start_point = (overlay_img.shape[1]//2,
               overlay_img.shape[0]//2)
end_point = (int(overlay_img.shape[1]/2 - gravity_in_RGB[0]*scale),
             int(overlay_img.shape[0]/2 + gravity_in_RGB[1]*scale))

[cv2.circle(overlay_img, tuple(img_points[idx, :].astype(int)), radius=10,
            color=(255, 255, 0), thickness=-1) for idx in range(img_points.shape[0])]
cv2.drawFrameAxes(overlay_img, cameraMatrix=RGB_intrinsic, distCoeffs=RGB_distortion,
                  rvec=rvecs, tvec=tvecs, length=0.03, thickness=5)
cv2.arrowedLine(overlay_img, start_point,
                end_point, color, thickness)
cv2.namedWindow("Board with gravity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Board with gravity", 960, 540)
cv2.imshow("Board with gravity", overlay_img)
cv2.waitKey(0)
