# Use code from https://github.com/alopezgit/DESC/issues/3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import cv2
from skspatial.objects import Line, Points
from skspatial.plotting import plot_3d

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


#   Function to select image point in console and store data points for fitting.
# def select_points(event, x, y, flags, param):
#     global points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#     return points

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
plt.colorbar(mapper)
plt.xlabel('pixel along column (v)')
plt.ylabel('pixel along row (u)')
plt.title('depth image')
plt.show()

## STORE DEPTH COORDINATES AS 3D ARRAY
U, V = np.meshgrid(np.array([list(range(1, depth_map.shape[0]+1))]),
                   np.array([list(range(1, depth_map.shape[1]+1))]))
test = np.array([U.reshape(-1, 1), V.reshape(-1, 1), depth_map.reshape(-1, 1)]).T.reshape(-1, 3)

##  2. FITTING VECTOR TO DEPTH DATA OF PLUMB BOB
data = np.array([[188.20114739629298, 109.55320892699535],
                 [188.20114739629298, 111.24782499054345],
                 [188.20114739629298, 113.42661707224815],
                 [188.20114739629298, 115.84749716303116],
                 [188.4432354053713, 117.54211322657926],
                 [188.4432354053713, 120.68925734459717],
                 [188.4432354053713, 122.86804942630187],
                 [188.4432354053713, 124.56266548984996]])

u_data = data[:, 1]
v_data = data[:, 0]
z_data = np.array([depth_map[np.ceil(u_data[idx]).astype(int),
                             np.ceil(v_data[idx]).astype(int)] for idx in range(data.shape[0])]).reshape(-1, 1)

data = np.hstack((data, z_data))

plt.imshow(colormapped_im)
plt.colorbar(mapper)
plt.scatter(v_data, u_data, s=20., color='black')
plt.xlabel('pixel along column (v)')
plt.ylabel('pixel along row (u)')
plt.title('depth image')
plt.show()

#   FIT WITH LINE FITTING THROUGH 3D POINTS


#   FIT WITH CURVE FITTING
#   We are fitting a straight line, so need a linear function
def linear_function(x, m, c):
    return m * x + c


#  Fit data to a curve using non-linear squares fit
popt_u, _ = scipy.optimize.curve_fit(linear_function, v_data, u_data)
fitted_u = linear_function(v_data, *popt_u)

popt_v, _ = scipy.optimize.curve_fit(linear_function, u_data, v_data)
fitted_v = linear_function(u_data, *popt_v)

#  Visualise fitted line
plt.imshow(colormapped_im)
plt.scatter(v_data, u_data, s=20., color='black', label='selected points')

# arrow_tail_idx = np.argmin(fitted_u)
# arrow_head_idx = np.argmax(fitted_u)
#
# plt.arrow(v_data[arrow_tail_idx], fitted_u.min(),
#       (v_data[arrow_head_idx] - v_data[arrow_tail_idx]),
#       (fitted_u.max() - fitted_u.min()),
#       width=1.2, color='pink', label='fitted vector (fit u)')

arrow_tail_idx = np.argmin(u_data)
arrow_head_idx = np.argmax(u_data)

# Vector information
start_point = [fitted_v[arrow_tail_idx], u_data.min(), z_data[arrow_tail_idx]]
dx = fitted_v[arrow_head_idx] - fitted_v[arrow_tail_idx]
dy = u_data.max() - u_data.min()
dz = z_data[arrow_head_idx] - z_data[arrow_tail_idx]
magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
theta = np.rad2deg(np.arcsin(dz/magnitude))
gamma = np.rad2deg(np.arcsin(dx/magnitude))

plt.arrow(start_point[0], start_point[1], dx, dy, linestyle='-', width=1.,
          alpha=0.7, color='orange', label='fitted vector (fit v)')

plt.title('Identified direction of gravity vector in depth image')
plt.xlabel('pixel along column (v)')
plt.ylabel('pixel along row (u)')
plt.legend()
plt.show()
