import cv2          # OpenCV version was 4.8
import cv2.aruco as aruco
import numpy as np
import pandas as pd

time = '2023-07-12_10-57-32'

#   Load camera intrinsic and extrinsic matrices
extrinsics = pd.read_csv('data/matrices/extrinsics_{}.csv'.format(time), sep=',', header=None).values.T
intrinsics = pd.read_csv('data/matrices/intrinsics_{}.csv'.format(time), sep=',', header=None).values.T
world_to_pv = pd.read_csv('data/matrices/world_to_pv_{}.csv'.format(time), sep=',', header=None).values.T
world_to_lt = pd.read_csv('data/matrices/world_to_lt_{}.csv'.format(time), sep=',', header=None).values.T

rotation = world_to_pv[:-1, :-1]
translation = world_to_pv[:-1, -1]
projection = np.matmul(intrinsics[:-1, :-1], rotation)

#   Initialize the ArUco parameters
aruco_params = aruco.DetectorParameters()
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

#   Store image points
# obj_pts = np.array([[105., 45., 1.], [105., 82., 1.], [105., 112., 1.],
#                     [75., 82., 1.], [75., 112., 1.]])
img_pts = np.array([[872., 584., 1.], [920., 593., 1.], [864., 631., 1.], [918., 637., 1.], [775., 567., 1.]])
aruco_pts = []
aruco_ids = []

#   [u,v,1] = K*R*[x,y,z] + t
test = img_pts[-1] - translation
world_pts = np.matmul(np.linalg.inv(projection), test)
print('world coordinate of ID 6:', world_pts)

#   Load calibration depth images taken by HoloLens 2
img = cv2.imread('data/rgb_images/rgb_{}.png'.format(time), 1)

# Check for AruCo dictionary
for (arucoName, arucoDict) in ARUCO_DICT.items():
    aruco_dict = aruco.getPredefinedDictionary(arucoDict)

    # Detect the markers in the image
    corners, ids, rejected = aruco.ArucoDetector(aruco_dict, aruco_params).detectMarkers(img)

    # Draw detected markers on the image
    aruco.drawDetectedMarkers(img, corners, ids)

    # Print the IDs and corner coordinates of the detected markers
    if ids is not None:
        for i in range(len(ids)):
            # Calculate centre point
            marker_corners = corners[i][0]
            centre = np.mean(marker_corners, axis=0)

            # Store image points
            aruco_pts.append(centre)
            aruco_ids.append(ids[i])

            # Display important info
            print('---------')
            print("Marker ID: {}, {}".format(ids[i], arucoName))
            #print("Marker Corners:", corners[i])
            print("Marker Centre:", centre)

            # Draw a circle at the center point
            cv2.circle(img, tuple(centre.astype(int)), 3, (0, 255, 0), -1)

        # Display the image with detected markers
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



