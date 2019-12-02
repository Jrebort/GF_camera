import numpy as np
import sys
from decimal import Decimal

from list_process import TransData

#Generating new coordinate of specified keypoint.
#And calculate the distance between new pose and recommend pose.
class GenGuide:
    
    def gen_guide(self,_dis,recmd_body_group,coor_point,points_group):
        #if coor_point exist, generate the guiding of current pose.
        if coor_point!=0:
            #generate predefined coordinate by shifing the position of right wrist.
            #new_coor: list of tuple '[(x,y),...,(x,y)]',which length=8
            new_coor = self.gen_new_coor(coor_point)

            #Ensuring the RWrist is not at the boundary of frame.
            if new_coor == False:
                pass
            else:
                #replace the 4-th element of human_list with new_coor[i]
                #Newly assembled keypoints group: new_body (2-d list), list_length=8
                #new_body:[[x,y,...,x,y],...,[x,y,...,x,y]]
                new_body = []
                for i in range(len(new_coor)):
                    points_group[4] = new_coor[i]
                    obj_class_trans = TransData()
                    new_body.append(obj_class_trans.process(points_group))

                #calculate the distance between each new group of keypoints and recommend pose keypoints.
                list_dis = self.cal_new_dis(recmd_body_group,new_body)
                min_dis = min(list_dis)

                #while new generated pose get closer with recmd_body, show the guiding arrowedline.
                #arrowedline:direction from current position to new position of right wrist.
                if min_dis < _dis:
                    dst_coor = new_coor[list_dis.index(min(list_dis))]
                    return dst_coor
                else:
                    return False
        else:
            return False
    
    def gen_new_coor(self,old_cor):
        tup_list = []
        if (old_cor[0]-40)<=1 or (old_cor[0]+40)>=639 or (old_cor[1]-40)<=1 or (old_cor[1]+40)>=479:
            return False
        Lmove = (old_cor[0]-40,old_cor[1])
        Rmove = (old_cor[0]+40,old_cor[1])
        Umove = (old_cor[0],old_cor[1]-40)
        Dmove = (old_cor[0],old_cor[1]+40)
        LUmove = (old_cor[0]-40,old_cor[1]-40)
        RUmove = (old_cor[0]+40,old_cor[1]-40)
        LDmove = (old_cor[0]-40,old_cor[1]+40)
        RDmove = (old_cor[0]+40,old_cor[1]+40)
        tup_list.extend([Lmove,Rmove,Umove,Dmove,LUmove,RUmove,LDmove,RDmove])
        #tup_list.extend([Lmove,Rmove,Umove,Dmove])
        return tup_list
    
    def cal_new_dis(self,rcmd_body,new_body):
        #return distance list of rcmd_body and multi-new_body
        #rcmd_body:['imgID',(,),...,(,)], length=1~19
        #new_body:[[(,),...,(,)],[...],...,[...]], length=8
        list_dis = []
        vector1 = []    
        tmp_1 = rcmd_body[1:]   #tmp_1:[(,),...,(,)],length=1~18
        for j in range(len(tmp_1)):
            vector1.append(Decimal(tmp_1[j][0]))
            vector1.append(Decimal(tmp_1[j][1]))
        for i in range(len(new_body)): 
            vector2 = new_body[i]
            _d = abs(len(vector1)-max(len(vector1),len(vector2)))
            _e = abs(len(vector2)-max(len(vector1),len(vector2)))
            if len(vector1)-max(len(vector1),len(vector2))<0:
                for i in range(_d):
                    vector1.append(0)
            elif len(vector2)-max(len(vector1),len(vector2))<0:
                for i in range(_e):
                    vector2.append(0)

            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            _dis = np.sqrt(np.sum(np.square(vector1-vector2)))
            list_dis.append(_dis)
        return list_dis
