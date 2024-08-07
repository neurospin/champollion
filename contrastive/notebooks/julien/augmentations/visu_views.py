import numpy as np
import os
import dico_toolbox as dtx
import anatomist.headless as anatomist
from soma import aims
import io
import logging

import matplotlib.pyplot as plt
from numpy import int16

from contrastive.utils.plots.visu_utils import buffer_to_image

logger = logging.getLogger(__name__)


a = None
win = None

model_dir = '/volatile/jl277509/Runs/02_STS_babies/Output/2024-08-06/16-59-30_44'


class Visu_Anatomist:

    def __init__(self, ):
        global a
        global win
        a = anatomist.Anatomist()
        win = a.createWindow('3D')
        win.setHasCursor(0)

    def plot_bucket(self, arr, buffer):
        """Plots as 3D buckets

        Args:
            arr : [size_X, size_Y, size_Z]
            buffer (boolean): True -> returns PNG image buffer
                            False -> plots the figure
        """
        global a
        global win
        vol = aims.Volume(arr.astype(int16))
        bucket_map = dtx.convert.volume_to_bucketMap_aims(vol)
        bucket_a = a.toAObject(bucket_map)
        bucket_a.addInWindows(win)
        view_quaternion = [0.4, 0.4, 0.5, 0.5]
        win.camera(view_quaternion=view_quaternion)
        win.imshow(show=False)

        if buffer:
            win.removeObjects(bucket_a)
            return buffer_to_image(buffer=io.BytesIO())
        else:
            plt.show()

views_dir = os.path.join(model_dir, 'logs/views/')
view1 = np.load(views_dir+'view1.npy')
view2 = np.load(views_dir+'view2.npy')

visu_anatomist = Visu_Anatomist()

arr = view1[0][0]
image = visu_anatomist.plot_bucket(arr, buffer=True)

input('Press a key to continue')