import os.path
import glob
import scipy.misc

def training_visualization():
    visualization.remove_if_exists('Graph/*')

def remove_if_exists(pattern):
    for f in glob.glob(pattern):
        os.remove(f)

def load_image(path):
    im = scipy.misc.imread(path, flatten=False, mode='L') # L is (8-bit pixels, black and white)
    return im