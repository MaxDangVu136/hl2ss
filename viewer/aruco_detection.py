import cv2                      # OpenCV version was 4.8
import cv2.aruco as aruco
import numpy as np
import glob

# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates on the Shell
        print(x, ' ', y)

        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        point = (x, y)
        cv2.circle(img, point, 3, (0, 255, 0), -1)
        cv2.putText(img, str(x) + ',' +
                    str(y), point, font,
                    1, (0, 255, 0), 1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the Shell
        print(x, ' ', y)

        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


#   Initialize the ArUco parameters
aruco_params = aruco.DetectorParameters()
ARUCO_DICT = {
    # "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    # "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    # "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    # "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    # "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    # "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    # "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    # "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    # "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    # "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    # "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    # "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

#   ChArUco board details
[rows, columns] = [6, 6]

#   Load depth images
files = glob.glob('data/rgb_images/*.png')

#   3D object points
objp = np.zeros((int(rows*columns), 3))
objp[:, :2] = np.mgrid[0:columns, 0:rows].reshape(-1, 2)

#   Store object and image points
obj_pts = []
img_pts = []

#   Load calibration depth images taken by HoloLens 2
for image in files:
    img = cv2.imread(image, 1)

    # Check for AruCo dictionary
    for (arucoName, arucoDict) in ARUCO_DICT.items():
        aruco_dict = aruco.getPredefinedDictionary(arucoDict)

        # Detect the markers in the image
        corners, ids, rejected = aruco.ArucoDetector(aruco_dict, aruco_params)\
                                .detectMarkers(img)

        # Draw detected markers on the image
        aruco.drawDetectedMarkers(img, corners, ids)

        # Print the IDs and corner coordinates of the    detected markers
        if ids is not None:
            for i in range(len(ids)):

                # Calculate centre point
                marker_corners = corners[i][0]
                centre = np.mean(marker_corners, axis=0)

                # Store object and image points
                obj_pts.append(objp)
                img_pts.append(corners)

                # Display important info
                print('---------')
                print("Marker ID: {}, {}".format(ids[i], arucoName))
                print("Marker Corners:", corners[i])
                print("Marker Centre:", centre)

                # Draw a circle at the center point
                cv2.circle(img, tuple(centre.astype(int)), 3, (0, 255, 0), -1)

            # Display the image with detected markers
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

#   HoloLens cam calibration
# ret, cam, dist, r, t = cv2.calibrateCamera(
#                         obj_pts, img_pts, gray.shape[::-1], None, None)

