import numpy as np
from decimal import Decimal

class DataSetProcess:

    def read_txt(self,path_txt_file):
        f = open(path_txt_file,"r")
        data = []
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line)
        f.close()
        return data

    def write_txt(self,path_txt_file,content):
        with open(path_txt_file,"a+") as f:
            for i in range(len(content)):
                f.writelines(str(content[i]))
                if i < len(content)-1:
                    f.writelines("\n")

    def trans_original(self,data):
        for i in range(len(data)):
            data[i] = data[i].replace('(','')
            data[i] = data[i].replace(')','')
            data[i] = data[i].replace(',','')
        return data

    def normalization(self,data):
        data = np.array(data,dtype = 'float_')
        _range = np.max(data) - np.min(data)
        res = (data - np.min(data)) / _range
        res = res.astype(float).tolist()
        for i in range(len(res)):
            a = Decimal(res[i])
            res[i] = round(a,5)
        return res

    def cal_relative_position(self,vector_x,vector_y):
        col_1 = []
        col_2 = []
        center_point_x = vector_x[1]
        center_point_y = vector_y[1]
        for i in range(len(vector_x)):
            col_1.append(vector_x[i] - center_point_x)
            col_2.append(vector_y[i] - center_point_y)
        return col_1,col_2

    def write_txt_2(self,path_txt_file,content,mode):
        with open(path_txt_file,"a+") as f:
            if mode==1:
                f.writelines(str(content))
                f.writelines("\n")
            elif mode==2:
                f.writelines("\n")
            elif mode==3:
                f.writelines(str(content))
                f.writelines(" ")

    def function_2(self,data,path_txt_file):
        X,Y = [],[]
        for i in range(len(data)):
            bol = ' ' in data[i]
            #save the imgID to .txt file
            if data[i]!='' and bol==False:
                self.write_txt_2(path_txt_file,data[i],mode=1)
                continue
            #normalization and cal relative positions
            if data[i]!='' and bol==True:
                X.append(data[i].split(' ')[0])
                Y.append(data[i].split(' ')[1])
            #package one body which will be transformed with normalization and relative cal.
            if data[i] == '':
                col_1 = self.normalization(X)
                col_2 = self.normalization(Y)
                #cal the relative position of key point
                col_1,col_2 = self.cal_relative_position(col_1,col_2)
                for j in range(len(col_1)):
                    self.write_txt_2(path_txt_file,col_1[j],mode=3)
                    self.write_txt_2(path_txt_file,col_2[j],mode=1)
                self.write_txt_2(path_txt_file,None,mode=2)
                X.clear()
                Y.clear()
