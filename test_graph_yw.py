# test
from datetime import datetime
import os
import sys
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import get_graph

# from yolov3_tf2.models_ref import YoloV3
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# sys.argv=['']
FLAGS(sys.argv)
# FLAGS.flags_into_string()
yolo = YoloV3(classes=80) # might need to run twice


# weights_file = './data/yolov3.weights'
# with open(weights_file, "rb") as fp:
#     _ = np.fromfile(fp, dtype=np.int32, count=5)
#     weights = np.fromfile(fp, dtype=np.float32)

# # name_scope [ok]
# inputs = tf.keras.Input(shape=[2])
# with get_graph().as_default(), tf.name_scope('block'):
#     outputs = tf.keras.layers.Dense(10)(inputs)
# model = tf.keras.Model(inputs, outputs)
# for w in model.weights:
#     print(w.name)

# tensorboard
# logdir = os.path.join("logs", datetime.now().strftime('%Y%m%d-%H%M%S'))
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True, profiler=False)
# with writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0)