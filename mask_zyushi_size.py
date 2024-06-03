import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path


# kentai = ['hansyaban']

"""remove wavelength"""
#削除したい波長
rem_wave =[652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904] 

wave_length = [652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
              970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
              1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
              1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
              1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
              1552,1558,1564,1570,1576,1582,1588,1594,1600]
del_wave_length = wave_length.copy()
for rem in rem_wave:
    #print(rem)
    del_wave_length.remove(rem)
print(del_wave_length)
total_l = len(wave_length)
l = len(wave_length) - len(rem_wave)
                
                
def main():   
            # for wl in del_wave_length:
            #     print('=====')
            #     print(wl)
            # mask0 = cv2.imread(r'20230901_fiber\mask\{}\2004.png'.format(2), flags=cv2.IMREAD_GRAYSCALE)
            # mask2 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(2), flags=cv2.IMREAD_GRAYSCALE)
            # mask4 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(4), flags=cv2.IMREAD_GRAYSCALE)
            # mask6 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(6), flags=cv2.IMREAD_GRAYSCALE)
            # mask7 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(7), flags=cv2.IMREAD_GRAYSCALE)
            # mask9 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(9), flags=cv2.IMREAD_GRAYSCALE)
            # mask11 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(11), flags=cv2.IMREAD_GRAYSCALE)
            # mask14 = cv2.imread(r'20230901_fiber\mask\{}\2005.png'.format(14), flags=cv2.IMREAD_GRAYSCALE)               

            # a0,b0 = np.where(mask0==255)
            # a2,b2 = np.where(mask2==255)
            # a4,b4 = np.where(mask4==255)
            # a6,b6 = np.where(mask6==255)
            # a7,b7 = np.where(mask7==255)
            # a9,b9 = np.where(mask9==255)
            # a11,b11 = np.where(mask11==255)
            # a14,b14 = np.where(mask14==255)

            # print(mask0.shape,(1024*1280), len(a0), len(a2), len(a4), len(a6), len(a6), len(a7), len(a9), len(a11), len(a14))

            n0 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(0), delimiter=",")[:,:]
            n2 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(2), delimiter=",")[:,:]
            n4 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(4), delimiter=",")[:,:]
            n6 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(6), delimiter=",")[:,:]
            n7 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(7), delimiter=",")[:,:]
            n9 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(9), delimiter=",")[:,:]
            n11 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(11), delimiter=",")[:,:]
            n14 = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(14), delimiter=",")[:,:]

            print(n0.shape, len(n0), len(n2), len(n4), len(n6), len(n7), len(n9), len(n11), len(n14))




"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))