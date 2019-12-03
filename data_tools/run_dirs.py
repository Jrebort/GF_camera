"""
images => result.txt
"""
import argparse
import logging
import time
import glob
import ast
import os
import dill

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh



logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='pose_images\\eg_motion\\')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large', help=' mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    files_grabbed = glob.glob(os.path.join(args.folder, '*.jpg'))
    result = open('result_eg.txt', 'w')
    #pictures = open('pictures.txt', 'w')
    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image !
        
        filename = file.replace('pose_images\\eg_motion\\', '')
        logger.info('filename'+filename)
        
        result.write(filename+'\n')
        
        image = common.read_imgfile(file, None, None)
        h,w,c = image.shape
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        centers = {}
        x = []
        y = []
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                centers[i] = center
                x.append(center[0])
                y.append(center[1])
                result.write(str(center)+'\n')
            result.write('\n')
            break
        elapsed = time.time() - t
        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))