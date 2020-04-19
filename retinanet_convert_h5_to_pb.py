# import keras
import keras

# import keras retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from keras import backend as K

def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Converting Keras .h5 model to Tensorflow .pb model')       
    
    parser.add_argument('--backbone',    help='The backbone network model name', type=str, default='resnet50')

    parser.add_argument('input',       help='Input Keras .h5 model file name', type=str, default=None)
    parser.add_argument('output',        help='Output Tensorflow .pb model file name', type=str)

    return parser.parse_args()          

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

args = parse_args()

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model = models.load_model(args.input, backbone_name=args.backbone)

model.summary()

print(model.inputs)
print(model.outputs)

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

#tf.train.write_graph(frozen_graph, "models", "inference_omc_cell_detection.pb", as_text=False)
# Finally we serialize and dump the output graph to the filesystem
with tf.gfile.GFile(args.output, 'wb') as f:
    f.write(frozen_graph.SerializeToString())
print('%d ops in the final graph.' % len(frozen_graph.node))

#output_names=[out.op.name for out in model.outputs]

#print(output_names)