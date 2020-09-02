##
import tensorflow as tf
import glob
import model
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 총 몇개의 figure를 다 뽑아내는 걸로 합시다
#
models_RNN_static_dir = glob.glob('saved_model/ANN_x축_static*.hdf5')


##
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(nrows=2, # row 몇 개
                       ncols=1, # col 몇 개
                       height_ratios=[2, 1], width_ratios = [1]
                      )
#
