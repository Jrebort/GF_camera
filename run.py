import argparse
import logging
import time
import sys
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import cv2
import numpy as np

from list_process import TransData
from find_nearest import ComputeDistance
from gen_guide import GenGuide

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

def fun_load_param():
    parser = argparse.ArgumentParser(description='GFCamera real-time processing and display')
    parser.add_argument('--database', type=str, default='./data/result_3.txt')
    parser.add_argument('--poseimg', type=str, default='./pose_images/')
    parser.add_argument('--scaleX', type=float, default=0.24,help='width scaling ratio of pasted pose image')
    parser.add_argument('--scaleY', type=float, default=0.27,help='height scaling ratio of pasted pose image')
    parser.add_argument('--guidePT', type=int, default=4)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--tensorrt', type=str, default="False",help='for tensorrt process.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    #load params of model
    args = fun_load_param()

    #model initialization
    logger.debug('model initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    
    #camera initialization
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    
    #get frame size and calculate pasted image size '(w,h)'
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    paste_pose_size = (int(size[0]*args.scaleX),int(size[1]*args.scaleY))
    
    #read frame
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    #loop: read frame and display
    while True:

        ret_val, image = cam.read()

        #get inference results of human body estimation
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        human_list = []
        guidePT = 0

        #get only one group of human body keypoints.
        for human in humans:
            #traversal the keypoints groups but get only one group.
            for i in range(common.CocoPart.Background.value):
                #if can't detecte current keypoint, continue.
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                #reduction the detected keypoints and save as tuple:(x,y)
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                #remember the right Wrist coordinate
                if i == args.guidePT:
                    guidePT = center
                #save tuple of keypoint to list:[(x,y),...,(x,y)]
                human_list.append(center)
            break
        
        #self-customize class initialization
        #transform current detected keypoints group:
        ###### [(x,y),...,(x,y)] TO [x1,y1,x2,y2,...,xn,yn]
        ###### where x1~xn and y1~yn are normalized version.
        #return:
        ######## 'trans_ori_data': the transformed version of keypoints group.
        obj_class_trans = TransData()
        trans_ori_data = obj_class_trans.process(human_list)
        
        #load database and find the nearest pose.
        #return:
        ######## '_dis': the min distance
        ######## 'index_minValue': the index of min distance which used to get imgID
        obj_class_compute = ComputeDistance(args.database)
        load_data = obj_class_compute.fun_load_data()
        _dis,index_minValue = obj_class_compute.get_nearest(load_data,trans_ori_data)
        
        #guiding class initialization
        obj_class_guide = GenGuide()
        
        #get the nearest imgID
        imgID = load_data[index_minValue][0]

        #Get recmd_body coordinate group
        recmd_body_group = load_data[index_minValue]

        #Generating guiding of specified keypoint.
        res_guide = obj_class_guide.gen_guide(_dis,recmd_body_group,guidePT,human_list)
        if res_guide!=False:
            cv2.arrowedLine(image, guidePT, res_guide, (0,0,255),5,8,0,0.25)
        else:
            pass

        #render the recommend pose image to the bottom-right corner of frame.
        markImg = cv2.imread(args.poseimg+imgID)
        markImg = cv2.resize(markImg,paste_pose_size,interpolation=cv2.INTER_CUBIC)
        image = fun_paste_img(image,markImg,size,paste_pose_size)

        #draw real-time skeleton diagram on human body.
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,"FPS: %f" % (1.0 / (time.time() - fps_time)),(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.putText(image,"Score: %f"%(1.0 / float(_dis)),(500,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.imshow('GFCamera', image)

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
