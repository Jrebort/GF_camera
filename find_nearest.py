import numpy as np
import sys
from decimal import Decimal
import linecache

class ComputeDistance:
    
    path_txt_file = ''
    path_ori_file = ''
    current_point_group = []
    rows_num = 0 #the sum of database's rows number
    
    def __init__(self,p1,p2,data):
        self.path_txt_file = p1
        self.path_ori_file = p2
        self.current_point_group = data[:]
        _file = open(self.path_txt_file)
        self.rows_num = len(_file.readlines())
        _file.close()

    def read_txt(self,line_number):
        key_points = []

        file = open(self.path_txt_file,'r')
        
        for (num,value) in enumerate(file,line_number):
            #if cur line is imgID,continue
            value = linecache.getline(self.path_txt_file,num).strip()
            value = value.strip('\n')
            #if cur line is imgID,continue
            if value.find(' ')==-1 and len(value)!=0:
                key_points.append(value)
                continue
            #if cur line is point,save to list
            if value != '':
                #if the list is empty,record the beginning line number of the txt file.
                if len(key_points)==0:
                    key_points.append(num)
                    key_points.append(value)
                else:
                    key_points.append(value)
            #if cur line is empty,record line number to list and break
            elif value == '':
                key_points.append(num)
                break
        
        file.close()
        
        if len(key_points)<=1:
            print("you may sent a wrong line number.please check.")
            return
        
        return key_points
    
    def cal_body_distance(self,body_pre):
        tmp = []
        vector1 = []
        vector2 = self.current_point_group[:]
        for i in range(len(body_pre)):
            tmp = body_pre[i].split(' ')
            vector1.append(Decimal(tmp[0]))
            vector1.append(Decimal(tmp[-1]))
        d = abs(len(vector1)-max(len(vector1),len(self.current_point_group)))
        e = abs(len(self.current_point_group)-max(len(vector1),len(self.current_point_group)))
        if len(vector1)-max(len(vector1),len(self.current_point_group))<0:
            for i in range(d):
                vector1.append(0)
        elif len(self.current_point_group)-max(len(vector1),len(self.current_point_group))<0:
            for i in range(e):
                vector2.append(0)
        
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        return np.sqrt(np.sum(np.square(vector1-vector2)))
    
    def partition(self,arr,low,high): 
        i = ( low-1 )
        pivot = arr[high]     
        for j in range(low , high): 
            if   arr[j] <= pivot: 
                i = i+1 
                arr[i],arr[j] = arr[j],arr[i]   
        arr[i+1],arr[high] = arr[high],arr[i+1] 
        return ( i+1 ) 

    def quickSort(self,arr,low,high): 
        new_arr = arr
        if low < high: 
            pi = self.partition(new_arr,low,high) 
            self.quickSort(new_arr, low, pi-1) 
            self.quickSort(new_arr, pi+1, high)
        return new_arr
    
    def get_nearest(self):
        line_number = 1
        line_number_set = {}
        dis_set = []
        count = 0
        tmp_disset = []
        
        while line_number<=self.rows_num:
            
            points_group = self.read_txt(line_number)
            
            body_points = points_group[1:-1]
            
            #cal the distance between two body points' groups
            #record the caled distance in a list
            dis = self.cal_body_distance(body_points)
            dis_set.append(dis)
            line_number_set[count]=points_group[0]
            
            line_number = int(points_group[-1])+1
            count = count + 1
        
        #sort the recorded list,select the min value
        #get the min value line number
        
        tmp_disset = dis_set[:]
        
        sorted_list = self.quickSort(tmp_disset,0,len(dis_set)-1)
        
        print('The shortest dis is:',sorted_list[0])
        img_id = line_number_set[dis_set.index(sorted_list[0])]
        
        return img_id