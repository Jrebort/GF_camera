import numpy as np
import sys
from decimal import Decimal
import linecache

#To trans data '[(x0,y0),...,(xn,yn)]' to normalized view: '[x^0,y^0,...,x^n,y^n]'
#Flatten the tuple of list to 1-D list
#Transform original cooridinate to normalized view and calculate the relative position with 'center_point'.
class TransData:
    
    __data_pre = []
    __data = []
    
    def __init__(self):
        self.__data_pre = []
        self.__data = []
    
    def pre_process(self,data_ori):
        for i in range(len(data_ori)):
            self.__data_pre.append(data_ori[i][0])
            self.__data_pre.append(data_ori[i][1])
        pass
    
    def normalization(self):
        data_x = []
        data_y = []
        data = np.array(self.__data_pre,dtype = 'float_')
        for i in range(len(data)):
            if i%2==0:
                data_x.append(data[i])
            elif i%2!=0:
                data_y.append(data[i])
                
        if len(data_x)>0 and len(data_y)>0:
            _range_x = np.max(data_x) - np.min(data_x)
            _range_y = np.max(data_y) - np.min(data_y)
            res_x = (data_x - np.min(data_x)) / _range_x
            res_x = res_x.astype(float).tolist()
            res_y = (data_y - np.min(data_y)) / _range_y
            res_y = res_y.astype(float).tolist()
    
            for i in range(len(res_x)):
                a = Decimal(res_x[i])
                b = Decimal(res_y[i])
                res_x[i] = round(a,5)
                res_y[i] = round(b,5)
            pass
    
            res_x,res_y = self.cal_relative_position(res_x,res_y)
        
            for i in range(len(res_x)):
                self.__data.append(res_x[i])
                self.__data.append(res_y[i])
        else:
            pass

    def cal_relative_position(self,vector_x,vector_y):
        col_1 = []
        col_2 = []
        center_point_x = vector_x[1]
        center_point_y = vector_y[1]
        for i in range(len(vector_x)):
            col_1.append(vector_x[i] - center_point_x)
            col_2.append(vector_y[i] - center_point_y)
        return col_1,col_2
    
    #[Program call entry]#
    #Input: list of tuple: '[(x0,y0),...,(xn,yn)]'
    #Return: 1-D list: '[x^0,y^0,...,x^n,y^n]'
    def process(self,data_ori):
        self.pre_process(data_ori)
        self.normalization()
        return self.__data