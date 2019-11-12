import cv2
import os
from PIL import Image
import numpy as np

from dataset_process import DataSetProcess
from list_process import TransData
from find_nearest import ComputeDistance
from run_for_inference import BodyEstimation

class BackEndClass:
    
    
    path_res1 = ''
    path_res2 = ''
    path_res3 = ''
    path_input_video = ''
    path_output_video = ''
    path_pose_img = ''

    def __init__(self,path_res1,path_res2,path_res3,path_input_video,path_output_video,path_pose_img):
        self.path_res1 = path_res1
        self.path_res2 = path_res2
        self.path_res3 = path_res3
        self.path_input_video = path_input_video
        self.path_output_video = path_output_video
        self.path_pose_img = path_pose_img


    def fun_dataset_process(self):
        obj_dataset = DataSetProcess()
        obj_dataset.write_txt(self.path_res2,obj_dataset.trans_original(obj_dataset.read_txt(self.path_res1)))
        print('result 1 to 2 has been done.')
        obj_dataset.function_2(obj_dataset.read_txt(self.path_res2),self.path_res3)
        print('result 2 to 3 has been done.')
    
    def fun_paste_img(self,img1,img2):
        rows,cols,channels = img2.shape
        roi = img1[327:480, 512:640]
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

        img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

        dst = cv2.add(img1_bg,img2_fg)
        img1[327:480, 512:640] = dst
        return img1
        


    def fun_backend_main(self):
        obj_est = BodyEstimation()
        
        ################################################################
        vc = cv2.VideoCapture(self.path_input_video)
        size = (640,480)
        fps = 30
        videoWrite = cv2.VideoWriter(self.path_output_video,cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)
        numFramesRenaining = 10*fps - 1
        
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        #every timeF frame to save
        timeF = 10
        c = 1
        while rval and numFramesRenaining>1:
            if (c % timeF == 0) or c==1:
                #1.Finding the nearest pose-image of current frame;
                #2.Then paste the nearest pose-image to current frame;
                #3.Last,write the modified frame to a new video.
                human_list = obj_est.fun_main(frame)
                obj_class_trans = TransData(human_list)
                obj_class_compute = ComputeDistance(self.path_res3,self.path_res1,obj_class_trans.process())
                imgID = obj_class_compute.get_nearest()
                markImg = cv2.imread(self.path_pose_img+imgID)
                markImg = cv2.resize(markImg,(128,153),interpolation=cv2.INTER_CUBIC)
                add_frame = self.fun_paste_img(frame,markImg)
                videoWrite.write(add_frame)
                rval, frame = vc.read()
                #cv2.imwrite(pic_path + str(c) + '.jpg', frame)
            else:
                videoWrite.write(self.fun_paste_img(frame,markImg))
                rval,frame = vc.read()
            c = c + 1
            numFramesRenaining-=1
        vc.release()
        ################################################################
        
        print('Congratulations! Now you can watch the video in path: '+self.path_output_video)

##=#=#=#=#=#=#=#=#=#TEST#=#=#=#=#=#=#=#=#=#=#

if __name__ == '__main__':

    path_res1 = './data/result.txt'
    path_res2 = './data/result_2.txt'
    path_res3 = './data/result_3.txt'
    path_input_video = './data/test.avi'
    path_output_video = './data/out.avi'
    path_pose_img = './pose_images/'

    obj_backend = BackEndClass(path_res1,path_res2,path_res3,path_input_video,path_output_video,path_pose_img)
    
    #if you want to generate a new database,delete '#' in the begin of the next line.
    #obj_backend.fun_dataset_process()
    #[ATTENTION]
    #The condition above is you have estimated all img of your dataset and got a txt file including all body key points.
    #You can run the file 'run_for_inference.py' to generate new txt file and change the path of 'path_res1' to your own path.
    
    obj_backend.fun_backend_main()

##=#=#=#=#=#=#=#=#=#TEST#=#=#=#=#=#=#=#=#=#=#