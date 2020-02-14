import os
import sys
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3 # needed for define FLAGS.yolo_max_boxes
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

""" 
python tools/visualize_dataset.py \
    --classes data/voc2012.names \
    --dataset data/voc2012_train.tfrecord \
    --N 5 \
    --random False \
    --out_dir outputs/voc2012
    
python tools/visualize_dataset.py \
    --classes data/aop.names \
    --dataset data/aop_train.tfrecord \
    --N 5 \
    --random False \
    --out_dir outputs/aop
    
"""
flags.DEFINE_string('classes', 'data/voc2012.names', 'path to class names file')
flags.DEFINE_string('dataset', 'data/voc2012_train.tfrecord', 'path to dataset tfrecord')
flags.DEFINE_integer('N', 1, 'take N images')
flags.DEFINE_boolean('random', False, 'randomly take images or not')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('out_dir', 'outputs/voc2012', 'directory of output images')

# sys.argv = ['']
# FLAGS(sys.argv)

def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
        logging.info('created output directory: {}'.format(FLAGS.out_dir))
    dataset = load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    if FLAGS.random:
        dataset = dataset.shuffle(512)
    for ii, (image, labels) in enumerate(dataset.take(FLAGS.N)):
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if not (x1==0 and x2==0):
                boxes.append((x1, y1, x2, y2))
                scores.append(1)
                classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]
        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))
        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        output_path = os.path.join(FLAGS.out_dir, "out{}.jpg".format(ii))
        cv2.imwrite(output_path, img)
        logging.info('output saved to: {}'.format(output_path))


if __name__ == '__main__':
    app.run(main)
