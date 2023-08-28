import numpy as np
import pandas as pd
import open3d as o3d
import ChArUco_pose_est


#   Rigid base corners (board b)
rigid_corners_board = pd.read_csv('data/realsense/simulation_data/rigid_corners_board.csv',
                                  sep=',', header=None).values.T

#   Gravity vector (accelerometer a)
gravity = pd.read_csv('data/realsense/simulation_data/gravity.csv',
                      sep=',', header=None).values.reshape(-1, 1)
gravity_unit = gravity/np.linalg.norm(gravity)
gravity_h = np.vstack((gravity, [0.]))

#   Depth point cloud (depth d)
pc = o3d.io.read_point_cloud('data/realsense/simulation_data/depth_vertices.ply')
depth_points = np.asarray(pc.points)
depth_points_h = ChArUco_pose_est.homogeneous_vectors(depth_points, 'column').T

#   Mechanics model (undeformed and deformed)
model_nodes = pd.read_csv(
    'data/cantilever/model_nodes.csv', sep=',', header=None).values
model_corner_nodes = pd.read_csv(
    'data/cantilever/model_fixed_end_corners.csv', sep=',', header=None).values
deformed_nodes = pd.read_csv(
    'data/cantilever/deformed_model_nodes.csv', sep=',', header=None).values

undeformed_model_nodes = ChArUco_pose_est.reorder_model_node_coordinates(model_nodes, False)
undeformed_model_nodes_h = ChArUco_pose_est.homogeneous_vectors(undeformed_model_nodes, 'column').T

deformed_model_nodes = ChArUco_pose_est.reorder_model_node_coordinates(deformed_nodes, False)
deformed_model_nodes_h = ChArUco_pose_est.homogeneous_vectors(deformed_model_nodes, 'column').T

#   Transformations required
T_depth_accel_ext = pd.read_csv('data/realsense/transformations/T_depth_accel.csv',
                            sep=',', header=None).values
T_depth_accel = np.vstack((T_depth_accel_ext, [0., 0., 0., 1.]))
T_accel_board = pd.read_csv('data/realsense/transformations/T_accel_board.csv',
                            sep=',', header=None).values

T_deformed_undeformed = np.array([[-0.01, 0.79, -0.61, 0.015],
                                  [1.00, 0.01, 0.00, 0.015],
                                  [0.01, -0.61, -0.79, 0.00],
                                  [0., 0., 0., 1.]])

#   Note that 'model' here is the undeformed model.
model_corner_nodes = ChArUco_pose_est.reorder_model_node_coordinates(
    model_corner_nodes, True)
ChArUco_pose_est.create_point_cloud(
    'data/realsense/simulation_data/undeformed_model_corners.ply', model_corner_nodes)

#   Compute transformation between board and model spaces
rigid_corners_board_align = rigid_corners_board.T[:, :-1]
rotation, translation = ChArUco_pose_est.find_transformation(
    rigid_corners_board_align, model_corner_nodes)
T_model_board = np.vstack(
    (np.hstack((rotation, translation.reshape(-1, 1))), [0., 0., 0., 1.]))
T_board_model = np.linalg.inv(T_model_board)

#   Inferred transformations
T_accel_model = T_board_model @ T_accel_board
T_depth_model = T_accel_model @ T_depth_accel

#   Transform depth, accel and board points to model space
depth_in_model, depth_in_model_h = ChArUco_pose_est.transform_from_X_to_Y(
    T_depth_model, depth_points_h)
_, accel_in_model_h = ChArUco_pose_est.transform_from_X_to_Y(
    T_accel_model, gravity_h)
board_corners_in_model, board_corner_in_model_h = ChArUco_pose_est.transform_from_X_to_Y(
    T_board_model, rigid_corners_board)
deformed_in_undeformed, deformed_in_undeformed_h = ChArUco_pose_est.transform_from_X_to_Y(
        T_deformed_undeformed, ChArUco_pose_est.homogeneous_vectors(deformed_model_nodes, 'column').T)
pc.points = o3d.utility.Vector3dVector(depth_in_model*1000)

# Bounding box definitions
x_min, x_max = -1/1000, 31/1000
y_min, y_max = depth_points[:,1].min(axis=0), 31/1000
# z_min, z_max = 0/1000, depth_points[:,2].min(axis=0)

# Keep only points inside the bounding box
filtered_pc = np.array([pt for pt in depth_in_model if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max])

#   Export info and view in meshlab
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/cantilever_model.txt',
                              1000 * undeformed_model_nodes, ',')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/depth_data.txt',
                              1000 * depth_in_model, ',')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/rigid_base_corners.txt',
                              1000 * board_corners_in_model, ',')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/deformed_model.txt',
                              1000 * deformed_in_undeformed, ',')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/transformed_gravity.txt',
                              accel_in_model_h[:-1], ',')
ChArUco_pose_est.write_to_txt('data/realsense/simulation_data/gravity.txt',
                              gravity, ',')

ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/cantilever_model.ply",
                                    undeformed_model_nodes * 1000)
o3d.io.write_point_cloud("data/realsense/simulation_data/transformed_depth_points.ply",
                         pc, write_ascii=True)
ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/transformed_filtered_depth_points.ply",
                                    filtered_pc * 1000)
ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/transformed_rigid_corner.ply",
                                    board_corners_in_model * 1000)
ChArUco_pose_est.create_point_cloud("data/realsense/simulation_data/transformed_deformed_geometry.ply",
                                    deformed_in_undeformed * 1000)

print("DONE!")