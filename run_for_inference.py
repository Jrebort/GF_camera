import argparse
import logging
import time
import glob
import ast
import os
import dill
import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

#Estimated the input image and return a 1-Dim list.
class BodyEstimation:

    def fun_log_record(self):
        logger = logging.getLogger('TfPoseEstimator')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def fun_config(self):
        parser = argparse.ArgumentParser(description='tf-pose-estimation run by a frame')
        parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
        parser.add_argument('--model', type=str, default='mobilenet_v2_large', help=' mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')
        args = parser.parse_args()
        return args

    def fun_main(self,image):

        #set args
        args = self.fun_config()
        scales = ast.literal_eval(args.scales)
        w, h = model_wh(args.resolution)
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

        # estimate human poses from a single image !
        #read image
        #image = common.read_imgfile(file, None, None)
        h,w,c = image.shape
        t = time.time()
        #estimate
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        for human in humans:
            human_list = []
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                human_list.append(center)
            break

        elapsed = time.time() - t
        
        return human_list