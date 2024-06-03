import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools


# kentai = ['hansyaban']

"""remove wavelength"""
#削除したい波長
rem_wave =[652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904
              ] 

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

def MASKTEKIYOU(mask1):
    data = cv2.imread(r'20231113\ganma\2001.png', flags=cv2.IMREAD_GRAYSCALE)  
    data1 = data.copy()
    data1[:,:] = 0

    n_labels, labels = cv2.connectedComponents(data)

    a1,a2 = np.where(data!=255)
    mask1[a1,a2] = 0

    a1,a2 = np.where(mask1==255)
    l1 = labels[a1,a2]
    # print(sorted(l1, reverse=True))
    l1 = list(set(l1))
    # print(l1)
    # print(labels)
    for t in l1:
        print(t)
        a3,a4 = np.where(labels==t)
        data1[a3,a4]=255
    
    return data1

def main():   
            # for wl in del_wave_length:
            #     print('=====')
            #     print(wl)
            img = cv2.imread(r'20231113\label3\006\{}.png'.format(910), flags=cv2.IMREAD_GRAYSCALE)    
            zero = img.copy()
            zero[:,:] = 0
            zero1 = zero.copy()
            zero2 = zero.copy()
            zero3 = zero.copy()
            zero4 = zero.copy()

            plt.imshow(img)
            plt.show()

            #zyushi
            #2
            mask_poly1 = np.array([[728,694],[730,723],[778,710],[761,678]])
            mask_poly2 = np.array([[639,701],[649,731],[691,728],[677,695]])             
            mask_poly3 = np.array([[584,558],[596,595],[633,597],[613,558]])
            mask_poly4 = np.array([[555,459],[564,509],[605,496],[567,453]])

            img1 = img.copy()
            img2 = img.copy()
            img3 = img.copy()
            img4 = img.copy()

            img_mask1 = cv2.fillPoly(img1, [mask_poly1], 255)
            img_mask2 = cv2.fillPoly(img2, [mask_poly2], 255)
            img_mask3 = cv2.fillPoly(img3, [mask_poly3], 255)
            img_mask4 = cv2.fillPoly(img4, [mask_poly4], 255)
            
            mask1 = cv2.fillPoly(zero1, [mask_poly1], 255)
            mask2 = cv2.fillPoly(zero2, [mask_poly2], 255)
            mask3 = cv2.fillPoly(zero3, [mask_poly3], 255)
            mask4 = cv2.fillPoly(zero4, [mask_poly4], 255)


            cv2.imwrite(r'20231113\mask\{}.png'.format(1), img_mask1)
            cv2.imwrite(r'20231113\mask\{}.png'.format(2), mask1)
            cv2.imwrite(r'20231113\mask\{}.png'.format(21), img_mask2)
            cv2.imwrite(r'20231113\mask\{}.png'.format(22), mask2)
            cv2.imwrite(r'20231113\mask\{}.png'.format(31), img_mask3)
            cv2.imwrite(r'20231113\mask\{}.png'.format(32), mask3)
            cv2.imwrite(r'20231113\mask\{}.png'.format(41), img_mask4)
            cv2.imwrite(r'20231113\mask\{}.png'.format(42), mask4)

            ####################

            data1 = MASKTEKIYOU(mask1)
            data2 = MASKTEKIYOU(mask2)
            data3 = MASKTEKIYOU(mask3)
            data4 = MASKTEKIYOU(mask4)
          
            cv2.imwrite(r'20231113\mask\{}.png'.format(101), data1)
            cv2.imwrite(r'20231113\mask\{}.png'.format(121), data2)
            cv2.imwrite(r'20231113\mask\{}.png'.format(131), data3)
            cv2.imwrite(r'20231113\mask\{}.png'.format(141), data4)

 

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))