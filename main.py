import os
import tensorflow as tf
import utils

print(os.environ['LD_LIBRARY_PATH'])
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print("test")