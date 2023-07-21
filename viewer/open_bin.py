import os
import charset_normalizer as chardet
from charset_normalizer import from_path, detect

path = '../calibration/rm_depth_longthrow/'
files = os.listdir(path)

test = path + files[0]
results = from_path(test)
print(str(results.best()))

for bin in files:
    with open(path + bin, 'rb') as file:
        bin_data = file.read()
        # detect = chardet.detect(bin_data)
        # str_data = bin_data.decode(detect['encoding'])
        result = detect(bin_data)

        #rial = detect([path + bin])

        #print(str_data)
