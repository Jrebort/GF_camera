import numpy as np
import sys
from decimal import Decimal

class ComputeDistance:
    path_txt3 = ''
    current_point_group = []
    
    def __init__(self,path_txt3,current_point_group):
        self.path_txt3 = path_txt3
        self.current_point_group = current_point_group
    
    def fun_load_data(self):
        data = []
        group_data = []
        for value in open(self.path_txt3,"r"):
            value = value.strip('\n')
            #print(value)
            if value != '':
                #print(value)
                if value.find(' ')!=-1:
                    value = tuple(value.split(' '))
                group_data.append(value)
            if value == '':
                data.append(group_data)
                group_data = []
        return data
    
    def cal_body_distance(self,load_data):
        #Flatting current/dataset body group points to 1-Dim and calculate. distance between them.
        list_dis = []        
        for i in range(len(load_data)):
            tmp = []
            vector1 = []
            vector2 = self.current_point_group[:]
            tmp = load_data[i][1:]
            for j in range(len(tmp)):                
                vector1.append(Decimal(tmp[j][0]))
                vector1.append(Decimal(tmp[j][1]))
            
            _d = abs(len(vector1)-max(len(vector1),len(self.current_point_group)))
            _e = abs(len(self.current_point_group)-max(len(vector1),len(self.current_point_group)))
            if len(vector1)-max(len(vector1),len(self.current_point_group))<0:
                for i in range(_d):
                    vector1.append(0)
            elif len(self.current_point_group)-max(len(vector1),len(self.current_point_group))<0:
                for i in range(_e):
                    vector2.append(0)

            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            _dis = np.sqrt(np.sum(np.square(vector1-vector2)))
            list_dis.append(_dis)
        return list_dis.index(min(list_dis))
    
    def get_nearest(self,load_data):
        #load data from result_3.txt to Memory(List)
        #To reduce time of loading data, move the sentence to 'backend.py'
        #load_data = self.fun_load_data()
        index_minValue = self.cal_body_distance(load_data)
        return load_data[index_minValue][0]