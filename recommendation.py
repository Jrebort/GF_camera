import numpy as np
import sys
from decimal import Decimal

#Computing the distance between current pose and each pose in database.#
#Then, find the nearest pose and return the min distance and index of min distance.#
class ComputeDistance:
    path_txt3 = ''
    
    def __init__(self,path_txt3):
        self.path_txt3 = path_txt3
    
    def fun_load_data(self):
        data = []
        group_data = []
        for value in open(self.path_txt3,"r"):
            value = value.strip('\n')
            if value != '':
                if value.find(' ')!=-1:
                    value = tuple(value.split(' '))
                group_data.append(value)
            if value == '':
                data.append(group_data)
                group_data = []
        return data
    
    def cal_body_distance(self,load_data,current_point_group):
        #Flatting current/dataset body group points to 1-Dim and calculate. distance between them.
        list_dis = []        
        for i in range(len(load_data)):
            tmp = []
            vector1 = []
            vector2 = current_point_group[:]
            tmp = load_data[i][1:]
            for j in range(len(tmp)):                
                vector1.append(Decimal(tmp[j][0]))
                vector1.append(Decimal(tmp[j][1]))
            
            _d = abs(len(vector1)-max(len(vector1),len(current_point_group)))
            _e = abs(len(current_point_group)-max(len(vector1),len(current_point_group)))
            if len(vector1)-max(len(vector1),len(current_point_group))<0:
                for i in range(_d):
                    vector1.append(0)
            elif len(current_point_group)-max(len(vector1),len(current_point_group))<0:
                for i in range(_e):
                    vector2.append(0)

            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            _dis = np.sqrt(np.sum(np.square(vector1-vector2)))
            list_dis.append(_dis)
        return min(list_dis),list_dis.index(min(list_dis))
    
    #Input:
    ## 'load_data': 2-D list '[['imgID',(x0,y0),...,(xn,yn)],...,[...]]'
    ## 'current_point_group': 1-D list '[x^0,y^0,...,x^n,y^n]'
    #Return:
    ## 'min_dis': Decimal type, the min distance.
    ## 'index_minValue': int type, the index of min distance.
    def get_nearest(self,load_data,current_point_group):
        min_dis,index_minValue = self.cal_body_distance(load_data,current_point_group)
        return min_dis,index_minValue
