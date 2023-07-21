import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

path = 'data/point_clouds/ascii/depth_point-cloud_2023-06-29_17-48-29.ply'
pcd = o3d.io.read_point_cloud(path)
points = Points(np.asarray(pcd.points))
plane = Plane.best_fit(points, full_matrices=False)

plot_3d(
    points.plotter(c='g', s=50, depthshade=True),
    plane.plotter(alpha=0.2, lims_x=(-0.5, 0.5), lims_y=(-0.5, 0.5)),
)

plt.show()