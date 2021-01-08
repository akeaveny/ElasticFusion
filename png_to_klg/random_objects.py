import cv2
import glob
import numpy as np

#####################
#####################
np.random.seed(0)

object_idx = np.arange(1, 5+1, 1)
num_objects_per_clutter_scene = 4
scenes = 5
for scene in range(scenes):
    objects = np.random.choice(object_idx, size=int(num_objects_per_clutter_scene), replace=False)
    ndds = np.random.choice(objects, size=int(2), replace=False)
    print("Scene:{} contains Objects:{} & NDDS:{}".format(scene, np.sort(objects), np.sort(ndds)))