#------------------------------------------------------------------------------
# This script receives uncompressed video from the HoloLens front RGB camera 
# and plays it. The camera support various resolutions and framerates. See
# etc/hl2_capture_formats.txt for a list of supported formats. The default
# configuration is 1080p 30 FPS. The stream supports three operating modes:
# 0) video, 1) video + camera pose, 2) query calibration (single transfer).
# Press esc to stop. Note that the ahat stream cannot be used while the pv
# subsystem is on.
#------------------------------------------------------------------------------

from pynput import keyboard

import time
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.1.7"

# Port
port = hl2ss.StreamPort.PERSONAL_VIDEO

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Camera parameters
width     = 640
height    = 360
framerate = 15

# Video encoding profile
profile = hl2ss.VideoProfile.RAW

#------------------------------------------------------------------------------

hl2ss.start_subsystem_pv(host, port)

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss.download_calibration_pv(host, port, width, height, framerate)
    print('Calibration')
    print(data.focal_length)
    print(data.principal_point)
    print(data.radial_distortion)
    print(data.tangential_distortion)
    print(data.projection)
    print(data.intrinsics)
else:
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss.rx_raw_pv(host, port, hl2ss.ChunkSize.PERSONAL_VIDEO, mode, width, height, framerate, profile, 1, 'bgr24')
    client.open()

    stride = hl2ss.get_nv12_stride(width)

    while (enable):
        data = client.get_next_packet()

        print('Pose at time {ts}'.format(ts=data.timestamp))
        print(data.pose)

        cv2.imshow('Video', data.payload.image)
        cv2.waitKey(1)

    client.close()
    listener.join()

hl2ss.stop_subsystem_pv(host, port)
