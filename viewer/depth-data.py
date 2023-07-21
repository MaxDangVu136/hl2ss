import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

#   Depth image path
date = '2023-07-06'
path = 'data/depth_images/original/{}/'.format(date)
files = os.listdir(path)

#   Depth threshold
threshold = [400, 550]
filter = True

for image in files:
    png = io.imread(path + image)
    test = np.unique(png)
    plt.figure(figsize=(9,7))

    if filter:
        png_filtered = np.array([[0 if (depth < threshold[0] or depth > threshold[1]) else depth for depth in row] for row in png])
        plt.imshow(png_filtered, cmap='binary')

    else:
        plt.imshow(png, cmap='binary')

    plt.colorbar(label = 'Depth intensities')
    plt.title('Depth map {}'.format(image[12:-4]))
    #plt.savefig('data/depth_images/plots/plotted_' + image)
    plt.show()