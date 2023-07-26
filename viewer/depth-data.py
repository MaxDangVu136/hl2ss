# Use code from https://github.com/alopezgit/DESC/issues/3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import cv2

## 1. LOAD AND VISUALISE DEPTH IMAGES
#   Depth image path
depth_data_path = 'data/points/depth_2023-07-25_14-31-29.csv'
depth_img_path = 'data/depth_images/original/depth_image_2023-07-25_14-31-29.png'
depth_map = pd.read_csv(depth_data_path, sep=',', header=None).values

#   Create depth map mask and colour map
mask = (depth_map != 0)
disp_map = 1 / depth_map
colormap = 'jet'
vmax = np.percentile(disp_map[mask], 95)
vmin = np.percentile(disp_map[mask], 5)
normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap=colormap)
mask = np.repeat(np.expand_dims(mask, -1), 3, -1)
colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
colormapped_im[~mask] = 255


#   Function to select image point in console.
def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))


def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


#   Function to select image point in console and store data points for fitting.
# image = cv2.imread('path/to/your/image.jpg')
# cv2.namedWindow('Image')
# cv2.setMouseCallback('Image', select_points)
# points = []
# while True:
#     cv2.imshow('Image', image)
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27 or len(points) == 2:  # Press 'Esc' key to exit or select two points
#         break
#
# cv2.destroyAllWindows()

fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

plt.imshow(colormapped_im)
plt.xlabel('pixel along column (v)')
plt.ylabel('pixel along row (u)')
plt.colorbar(mapper)
plt.show()

##  2. FITTING VECTOR TO DEPTH DATA OF PLUMB BOB
data = np.array([[188.94545454545445, 109.94155844155844],
                 [187.6467532467532, 127.08441558441558],
                 [187.90649350649343, 112.79870129870127],
                 [187.90649350649343, 116.17532467532467],
                 [188.42597402597394, 121.88961038961037]])

u_data = data[:, 1]
v_data = data[:, 0]


#   We are fitting a straight line, so need a linear function
def linear_function(x, m, c):
    return m * x + c


#  Fit data to a curve using non-linear squares fit
popt, _ = scipy.optimize.curve_fit(linear_function, v_data, u_data)
fitted_u = linear_function(v_data, *popt)

#  Visualise fitted line
plt.imshow(colormapped_im)
plt.scatter(v_data, u_data, color='red', label='selected points')
plt.plot(v_data, fitted_u, linewidth=3, color='black', label='fitted vector')
# plt.arrow(v_data.min(), fitted_u.max(),
#           (v_data.max() - v_data.max()),
#           (fitted_u.max() - fitted_u.max()),
#           width = 0.1, color='black', label='arrow')
plt.xlabel('pixel along column (v)')
plt.ylabel('pixel along row (u)')
plt.legend()
plt.show()
