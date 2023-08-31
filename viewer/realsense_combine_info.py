import numpy as np
import pandas as pd
import open3d as o3d
import ChArUco_pose_est

def pc_correct_orient(file_in, file_out, T_fix):
    """

    :param file_in:
    :param file_out:
    :param T_fix:
    :return:
    """
    #   Invert y and z axes of point cloud
    pcd = o3d.io.read_point_cloud(file_in)
    verts = np.asarray(pcd.points)

    #   Apply transformation to correct point cloud orientation
    verts_h = ChArUco_pose_est.homogeneous_vectors(verts, 'column').T
    verts_transformed, _ = ChArUco_pose_est.transform_from_X_to_Y(T_fix, verts_h)

    #   Save updated point cloud
    pcd.points = o3d.utility.Vector3dVector(verts_transformed)
    o3d.io.write_point_cloud(file_out, pcd)


def point_cloud_points(file):
    """

    :param file:
    :return:
    """
    pc = o3d.io.read_point_cloud(file)
    points = np.asarray(pc.points)

    return points, pc


#   Save realsense point cloud to correct orientation
T_fix_pc = np.array([[1., 0., 0., 0.],
                     [0., -1., 0., 0.],
                     [0., 0., -1., 0.],
                     [0., 0., 0., 1.]])

data = np.loadtxt("data/cantilever/deformed_model_c1_9.2.txt")
ChArUco_pose_est.create_point_cloud("data/cantilever/deformed_model_c1_9.2.ply", data)

pc_correct_orient('data/realsense/point_cloud_depth.ply',
                  'data/realsense/point_cloud_depth_modified.ply',
                  T_fix_pc)

#   Rigid base corners (board b)
rigid_corners_board = pd.read_csv('data/realsense/simulation_data/rigid_corners_board.csv',
                                  sep=',', header=None).values.T

#   Gravity vector (accelerometer a)
gravity = pd.read_csv('data/realsense/transformations/gravity.csv',
                      sep=',', header=None).values.reshape(-1, 1)
gravity_unit = gravity/np.linalg.norm(gravity)
gravity_h = np.vstack((gravity, [0.]))

#   Depth point cloud (depth d)
depth_points, depth_pc = point_cloud_points('data/realsense/point_cloud_depth_modified.ply')
depth_points_h = ChArUco_pose_est.homogeneous_vectors(depth_points, 'column').T

# #   Mechanics model (undeformed and deformed)
# model_nodes = pd.read_csv(
#     'data/cantilever/model_nodes.csv', sep=',', header=None).values
# model_corner_nodes = pd.read_csv(
#     'data/cantilever/model_fixed_end_corners.csv', sep=',', header=None).values
# deformed_nodes = pd.read_csv(
#     'data/cantilever/deformed_model_nodes.csv', sep=',', header=None).values
#
# undeformed_model_nodes = ChArUco_pose_est.reorder_model_node_coordinates(model_nodes, False)
# undeformed_model_nodes_h = ChArUco_pose_est.homogeneous_vectors(undeformed_model_nodes, 'column').T
#
# deformed_model_nodes = ChArUco_pose_est.reorder_model_node_coordinates(deformed_nodes, False)
# deformed_model_nodes_h = ChArUco_pose_est.homogeneous_vectors(deformed_model_nodes, 'column').T

#   Transformations from RealSense
T_depth_accel_ext = pd.read_csv('data/realsense/transformations/T_depth_accel.csv',
                            sep=',', header=None).values
T_depth_accel = np.vstack((T_depth_accel_ext, [0., 0., 0., 1.]))
T_accel_board = pd.read_csv('data/realsense/transformations/T_accel_board.csv',
                            sep=',', header=None).values
# T_aligned_color_depth = pd.read_csv('data/realsense/transformations/T_aligned_color_depth.csv',
#                             sep=',', header=None).values
# T_aligned_color_depth = np.vstack((T_aligned_color_depth, [0., 0., 0., 1.]))

# #   Note that 'model' here is the undeformed model.
# model_corner_nodes = ChArUco_pose_est.reorder_model_node_coordinates(
#     model_corner_nodes, True)
# deformed_model_corners, _ = point_cloud_points('data/realsense/simulation_data/deformed_model_corners.ply')
#
# deformed_model_corners = np.array([deformed_model_corners[3,:],
#                             deformed_model_corners[2,:],
#                             deformed_model_corners[0,:],
#                             deformed_model_corners[1,:]])
#
# #   Transform deformed model nodes to undeformed model nodes via base corners
# R, t = ChArUco_pose_est.find_transformation(model_corner_nodes, deformed_model_corners)
# T_deformed_undeformed = np.vstack((np.hstack((R, t.reshape(-1, 1))), [0., 0., 0., 1.]))

#   Compute transformation between board and model spaces
rigid_corners_board_align = rigid_corners_board.T[:, :-1]
ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/rigid_base_corners.ply",
                                    rigid_corners_board_align)
# rotation, translation = ChArUco_pose_est.find_transformation(
#     model_corner_nodes, rigid_corners_board_align)
# T_board_model = np.vstack(
#     (np.hstack((rotation, translation.reshape(-1, 1))), [0., 0., 0., 1.]))

#   Inferred transformations
# T_accel_model = T_board_model @ T_accel_board
# T_depth_model = T_accel_model @ T_depth_accel
T_depth_board = T_accel_board @ T_depth_accel
T_depth_board[0,-1] = T_depth_board[0,-1] - 0.016

#   Transform depth, accel and board points to model space
#   NOTE: do I want to do this, we are adding errors to our source data??
# depth_in_model, depth_in_model_h = ChArUco_pose_est.transform_from_X_to_Y(
#     T_depth_model, depth_points_h)

# aligned_color_in_model, _ = ChArUco_pose_est.transform_from_X_to_Y(
#     T_aligned_color_depth_model, depth_in_model_h)

depth_in_board, depth_in_board_h = ChArUco_pose_est.transform_from_X_to_Y(
    T_depth_board, depth_points_h)
_, accel_in_board_h = ChArUco_pose_est.transform_from_X_to_Y(
    T_accel_board, gravity_h)
gravity_corrected = np.array([accel_in_board_h[0], -accel_in_board_h[1], -accel_in_board_h[2]])

# board_corners_in_model, board_corner_in_model_h = ChArUco_pose_est.transform_from_X_to_Y(
#     T_board_model, rigid_corners_board)
# deformed_in_undeformed, deformed_in_undeformed_h = ChArUco_pose_est.transform_from_X_to_Y(
#         T_deformed_undeformed, ChArUco_pose_est.homogeneous_vectors(deformed_model_nodes, 'column').T)

# filtered_pc, _ = point_cloud_points('data/realsense/simulation_data/filtered_depth_point_cloud_right.ply')
# ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/horizontal_beam_right.txt',
#                               filtered_pc/1000, ' ')

#   Export info and view in meshlab
# ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/cantilever_model.txt',
#                               1000 * undeformed_model_nodes, ' ')
# ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/rigid_base_corners.txt',
#                               1000 * board_corners_in_model, ' ')
# ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/deformed_model.txt',
#                               1000 * deformed_in_undeformed, ' ')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/gravity_board_corrected.txt',
                              gravity_corrected.T, ' ')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/depth_vertices.txt',
                              depth_in_board, ' ')

depth_pc.points = o3d.utility.Vector3dVector(depth_in_board)
o3d.io.write_point_cloud("data/realsense/simulation_data/depth_in_board.ply",
                         depth_pc, write_ascii=True)
# ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/transformed_rigid_corner.ply",
#                                     board_corners_in_model * 1000)
# ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/transformed_deformed_geometry.ply",
#                                     deformed_in_undeformed * 1000)

print("DONE!")