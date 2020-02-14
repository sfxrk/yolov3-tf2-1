import os
import inspect
import sys
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
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

TODO: put visualize_dataset.py to project root, then it can run,
load_tfrecord_dataset is updated. why?
FLAGS.random is set to False, but printed as True when running the script, why?

    
"""
flags.DEFINE_string('classes', 'data/aop.names', 'path to class names file')
flags.DEFINE_string('dataset', 'data/aop_train.tfrecord', 'path to dataset tfrecord')
flags.DEFINE_integer('N', 1, 'take N images')
flags.DEFINE_boolean('random', False, 'randomly take images or not')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('out_dir', 'outputs/aop', 'directory of output images')
flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image')

# sys.argv = ['']
# FLAGS(sys.argv)

def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
        logging.info('created output directory: {}'.format(FLAGS.out_dir))
    # todo: maybe due to __pycache__, check the load_tfrecord_dataset file location, due to branch?
    print("debug: ************", load_tfrecord_dataset.__code__.co_varnames)
    print("debug: *********", inspect.getfile(load_tfrecord_dataset))
    print("debug: ********* ", inspect.signature(load_tfrecord_dataset))
    print("debug: ********* ", FLAGS.random)
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
