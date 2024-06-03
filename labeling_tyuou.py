import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import math


# kentai = ['hansyaban']

"""remove wavelength"""
#削除したい波長
rem_wave =[430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,610,
              616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904] 
# rem_wave =[910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]
                

wave_length = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,610,
              616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
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
            for wl in wave_length:
                print('=====')
                print(wl)
                mask = cv2.imread(r'20230901_fiber\mask\2102.png', flags=cv2.IMREAD_UNCHANGED)
                img = cv2.imread(r'20230901_fiber\affine\2\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                # mask = cv2.imread(r'20231218_fiber_prop\ganma\8.png', flags=cv2.IMREAD_UNCHANGED)
                # img = cv2.imread(r'20231218_fiber_kaizoudo\affine\16\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                mask1 =mask.copy()
                mask1[:,:] = 0

                # 連結成分のラベリングを行う。
                n_labels, labels = cv2.connectedComponents(mask)
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                # ラベル数
                print("number of labels:", n_labels)

                # 平均化
                for i in range(n_labels-1):
                    i = i+1
                    print(i)
                    mask1[:,:] = 0

                    x = centroids[i,0]
                    y = centroids[i,1]
                    print(x,y)

                    cv2.circle(mask1, center=(int(x),int(y)), radius=2 , color=255, thickness=-1)  

                    # r = math.sqrt(stats[i,4]/(math.pi)*1/2)

                    # cv2.circle(mask1, center=(int(x),int(y)), radius=int(r) , color=255, thickness=-1) 

                    a1,a2 = np.where(labels==i)
                    b1,b2 = np.where(mask1==255)
                    p = img[b1,b2]
                    # TOP100の平均
                    # aver = np.mean(p)
                    # aver = img[int(y),int(x)]

                    #notave
                    aver = p

                    print(p)
                    labels = labels.astype(np.uint16)                 
                    labels[a1,a2] = aver

                cv2.imwrite(r'20230901_fiber\label2\2\tyou_no\{}.png'.format(wl), labels)
                # cv2.imwrite(r'20231218_fiber_kaizoudo\label\16\r1\{}.png'.format(wl), labels)

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))