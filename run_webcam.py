import argparse
import logging
import time

from PIL import Image

from list_process import TransData
from find_nearest import ComputeDistance

import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def fun_paste_img(img1,img2,frame_size,paste_img_size):
    left_top_pos = (int(frame_size[0]-paste_img_size[0]),int(frame_size[1]-paste_img_size[1]))
    rows,cols,channels = img2.shape
    roi = img1[left_top_pos[1]:frame_size[1], left_top_pos[0]:frame_size[0]]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    dst = cv2.add(img1_bg,img2_fg)
    img1[left_top_pos[1]:frame_size[1], left_top_pos[0]:frame_size[0]] = dst
    return img1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    paste_pose_size = (int(size[0]*0.24),int(size[1]*0.27))
    
    ret_val, image = cam.read()
    
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        human_list = []
        for human in humans:
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                human_list.append(center)
            break
        
        obj_class_trans = TransData(human_list)
        obj_class_compute = ComputeDistance('./data/result_3.txt',obj_class_trans.process())
        load_data = obj_class_compute.fun_load_data()
        imgID = obj_class_compute.get_nearest(load_data)
        markImg = cv2.imread('./images/'+imgID)
        markImg = cv2.resize(markImg,paste_pose_size,interpolation=cv2.INTER_CUBIC)
        image = fun_paste_img(image,markImg,size,paste_pose_size)
        
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
