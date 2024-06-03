#波長選択 Compovision

import numpy as np
from pathlib import Path
from PIL import Image
from itertools import product
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
import cv2
import glob
import datetime
from sklearn import preprocessing
import scipy.signal
import time
import seaborn as sns


"""1. Setting"""
"""Set all wavelength"""
# rem_wave =[430,436,442,448,454,460,466,472,478,484
#            ,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904] 

# wave_length = [496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,
#               910,916,922,928,934,940,946,952,958,964,
#               976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]
# wave_length = [910,916,922,928,934,940,946,952,958,964,
#               976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]            
# wave_length = [1066, 1072, 1090, 1258, 1372, 1378]
# wave_length = [1126, 1162, 1204, 1288, 1312, 1366, 1438, 1450, 1546]
# wave_length = [1312, 1372, 1450, 1546]


# del_wave_length = wave_length.copy()
# for rem in rem_wave:
#     #print(rem)
#     del_wave_length.remove(rem)
# print(del_wave_length)
# total_l = len(wave_length)
# l = len(wave_length) #- len(rem_wave)


# def remove_wavelength(arr,rem_wave,wave_length):
#     """波長を削除"""      
#     rem_list = []
#     for r in rem_wave:
#         rem_no = wave_length.index(r)
#         rem_list.append(rem_no)        
    
#     del_arr = np.delete(arr, rem_list, 1)
#     return del_arr

def nums(first_number, last_number, step=1):
    return range(first_number, last_number + 1, step) 

wavelength =[]
# for wl in nums(1,1024):
#     wavelength.append(wl)
# print(wavelength)   

wavelength.append(1)
for wl in nums(1,146):
    wl = wl*7+1
    wavelength.append(wl)
print(wavelength)    

def SNV (text):
    """SNV処理するよ"""
    sc = StandardScaler()
    f = text.T
    sc.fit(f)
    std = sc.transform(f)
    std2 = std.T
    std2 = std2[~np.isinf(std2).any(axis=1), :] #欠損値の除外
    return std2

def seaborn_plot(raw,name):
    pixel = raw.shape[0]
    a = np.array([[0,0]])
    for index,wl in enumerate(wavelength):
        #print(wl)
        aa = np.full((pixel,1), int(wl))
        bb = raw[:,index].reshape(pixel,1)
        aa = np.append(aa,bb,axis=1)
        a = np.append(a,aa,axis=0)
        #print(a)
    a = a[1:,:]
    columns = ['wavelength (nm)', name]
    df = pd.DataFrame(data=a, columns=columns)
    #print(df)
    sns.lineplot(x="wavelength (nm)", y=name, data=df, errorbar='sd')
    return

"""Savitzky-Golay"""
def SG(array):#array:2D行列
    print('SG shape :',array.shape)
    h, w = array.shape
    abs_ave = np.zeros((h, w))
    for m in range(h):
        abs_ave[m,:] = scipy.signal.savgol_filter(array[m,:], 5, 3, deriv=0)
        #print(n)
    return abs_ave



def main():
            s = ''
            package = [[0,2,4,6,7,9,11,14]]
            # t = 'top100'
            # t = 'top80'
            # t = 'top50'
            # t = 'top30'
            # t = 'top10'
            # t = 'bottom80'
            # t = 'bottom50'
            # t = 'bottom30'
            # t = 'bottom10'
            # t = 'middle80'
            # t = 'middle50'
            # t = 'middle30'
            # t = 'middle10'
            # t = 'not_ave'
            # t = 'tyuou_ari'
            # t = 'tyuou_nasi'


            # name = '20230901_fiber\3Dcube'

            for p in package:
                print(p)
                
                # path0 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/0/abs.csv'
                path2 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/2/abs.csv'
                path4 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/4/abs.csv'
                path6 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/6/abs.csv'
                path7 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/7/abs.csv'
                path9 = r'Hyperspectral_imaging_zyushi_03/henkango/3d/9/abs.csv'
                path11= r'Hyperspectral_imaging_zyushi_03/henkango/3d/11/abs.csv'
                path14= r'Hyperspectral_imaging_zyushi_03/henkango/3d/14/abs.csv'

                # ROIData0 = np.loadtxt(path0, delimiter=",")[:,:]
                ROIData2 = np.loadtxt(path2, delimiter=",")[:,:]
                ROIData4 = np.loadtxt(path4, delimiter=",")[:,:]
                ROIData6 = np.loadtxt(path6, delimiter=",")[:,:]
                ROIData7 = np.loadtxt(path7, delimiter=",")[:,:]
                ROIData9 = np.loadtxt(path9, delimiter=",")[:,:]
                ROIData11 = np.loadtxt(path11, delimiter=",")[:,:]
                ROIData14 = np.loadtxt(path14, delimiter=",")[:,:]

                # ROIData0 = SG(ROIData0)
                # ROIData2 = SG(ROIData2)
                # ROIData4 = SG(ROIData4)
                # ROIData6 = SG(ROIData6)
                # ROIData7 = SG(ROIData7)
                # ROIData9 = SG(ROIData9)
                # ROIData11 = SG(ROIData11)
                # ROIData14 = SG(ROIData14)

                # ROIData0 = remove_wavelength(ROIData0,rem_wave,wave_length)
                # ROIData2 = remove_wavelength(ROIData2,rem_wave,wave_length)
                # ROIData4 = remove_wavelength(ROIData4,rem_wave,wave_length)
                # ROIData6 = remove_wavelength(ROIData6,rem_wave,wave_length)
                # ROIData7 = remove_wavelength(ROIData7,rem_wave,wave_length)
                # ROIData9 = remove_wavelength(ROIData9,rem_wave,wave_length)
                # ROIData11 = remove_wavelength(ROIData11,rem_wave,wave_length)
                # ROIData14 = remove_wavelength(ROIData14,rem_wave,wave_length)

                if s == 'SNV':
                    ROIData = SNV(ROIData)
                    notROIData = SNV(notROIData)
                    fROIData = SNV(fROIData)
                    fnotROIData =SNV(fnotROIData)

                #sns.set_palette("prism_r")
                # palette = ["blue","lime","gold","cyan","red","magenta"]#bad
                # palette = ["blue","orange","lime","red","cyan","gold","magenta"]#6
                palette = ["blue","lime","red","cyan","gold","magenta"]#x
                # palette = ["gray","blue","lime","red","cyan","gold","magenta"]#7
                # palette = ["gray","blue","lime","gold","cyan","red","magenta"]#x
                # palette = "Spectral"
                # palette = ["blue","orange","green","red","purple","brown","pink","gray"]
                # sns.set_palette(palette,8)
                sns.set_palette(palette)
                # seaborn_plot(ROIData0,'Absorbance (a.u.)')
                # seaborn_plot(ROIData2,'Absorbance (a.u.)')
                seaborn_plot(ROIData4,'Absorbance (a.u.)')
                seaborn_plot(ROIData6,'Absorbance (a.u.)')
                # seaborn_plot(ROIData7,'Absorbance (a.u.)')
                # seaborn_plot(ROIData9,'Absorbance (a.u.)')
                seaborn_plot(ROIData11,'Absorbance (a.u.)')
                seaborn_plot(ROIData14,'Absorbance (a.u.)')
                #plt.title(d+'_' +n1+'_'+ n2+'_' + str(del_wave_length[0]) + '_' + str(del_wave_length[-1]) + '_'+s)
                #plt.legend(labels=['tumor',"",'normal',"",'tumor_fat',"",'normal_fat',""])
                if s != 'SNV':
                    plt.ylim(0,1)
                if s == 'SNV':
                    plt.ylim(-3,3)
                
                plt.savefig(r'Hyperspectral_imaging_zyushi_03/henkango/3d/raw-8_1.png')
                plt.show()


if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))