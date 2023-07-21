## Code source: https://github.com/dranjan/python-plyfile

import plyfile
import os

path = 'data/point_clouds/binary'
files = os.listdir(path)

for ply in files:
    if os.path.exists(path + '/../ascii/' + ply):
        pass
    else:
        data = plyfile.PlyData.read(path + '/' + ply)
        data.text = True
        data.write(path + '/../ascii/' + ply)
