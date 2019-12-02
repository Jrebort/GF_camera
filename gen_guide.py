import numpy as np
import sys
from decimal import Decimal

#Generating new coordinate of specified keypoint.
#And calculate the distance between new pose and recommend pose.
class GenGuide:
    
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