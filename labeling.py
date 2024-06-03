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
            # for wl in wave_length:
            for wl in rem_wave:
                print('=====')
                print(wl)
                mask = cv2.imread(r'20230901_fiber\mask\2102.png', flags=cv2.IMREAD_UNCHANGED)
                img = cv2.imread(r'20230901_fiber\affine\7\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                # mask = cv2.imread(r'20231218_fiber_prop\ganma\8.png', flags=cv2.IMREAD_UNCHANGED)
                # img = cv2.imread(r'20231218_fiber_kaizoudo\affine\01\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)

                # 連結成分のラベリングを行う。
                n_labels, labels = cv2.connectedComponents(mask)

                # ラベル数
                print("number of labels:", n_labels)
                # fig, ax = plt.subplots(figsize=(7, 7))
                # ax.imshow(labels)
                # plt.show()

                # ラベリングした座標を抽出
                # a1,a2 = np.where(labels==1)
                # # a1,a2 = np.where(img==0)
                # print(a1,a2) # array1は行、array2は列
                # p = img[a1,a2]
                # # p = p.remove(0)
                # print(p)
                # # 100の平均
                # aver = np.mean(p)
                # # TOP50の平均
                # p_sort = sorted(p, reverse=True)
                # p_top50 = p_sort[0:math.floor(len(p_sort)/2)]
                # print(p_top50, len(p_top50), len(p))
                # aver_top50 = np.mean(p_top50)
                # # TOP30の平均
                # p_sort = sorted(p, reverse=True)
                # p_top30 = p_sort[0:math.floor(len(p_sort)*3/10)]
                # print(p_top30, len(p_top30), len(p))
                # aver_top30 = np.mean(p_top30)
                # # TOP10の平均
                # p_sort = sorted(p, reverse=True)
                # p_top10 = p_sort[0:math.floor(len(p_sort)/10)]
                # print(p_top10, len(p_top10), len(p))
                # aver_top10 = np.mean(p_top10)
                # print(aver, aver_top50, aver_top30, aver_top10)            
                # labels = labels.astype(np.uint16)
                # labels[a1,a2] = aver
                # print(labels)

                # 平均化
                for i in range(n_labels-1):
                    i = i+1
                    print(i) 
                    a1,a2 = np.where(labels==i)
                    p = img[a1,a2]
                    #not_ave
                    aver = p
                    # TOP100の平均
                    # aver = np.mean(p)
                    # TOP80の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top80 = p_sort[0:math.floor(len(p_sort)*8/10)]
                    # aver= np.mean(p_top80)
                    # TOP50の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top50 = p_sort[0:math.floor(len(p_sort)/2)]
                    # aver= np.mean(p_top50)
                    # TOP30の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top30 = p_sort[0:math.floor(len(p_sort)*3/10)]
                    # aver= np.mean(p_top30)
                    # TOP10の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top10 = p_sort[0:math.floor(len(p_sort)/10)]
                    # aver = np.mean(p_top10)
                    # BOTTOM80の平均
                    # p_sort = sorted(p)
                    # p_bottom80 = p_sort[0:math.floor(len(p_sort)*8/10)]
                    # aver= np.mean(p_bottom80)
                    # print(p)
                    # BOTTOM50の平均
                    # p_sort = sorted(p)
                    # p_bottom50 = p_sort[0:math.floor(len(p_sort)/2)]
                    # aver= np.mean(p_bottom50)
                    # BOTTOM30の平均
                    # p_sort = sorted(p)
                    # p_bottom30 = p_sort[0:math.floor(len(p_sort)*3/10)]
                    # aver= np.mean(p_bottom30)
                    # BOTTOM10の平均
                    # p_sort = sorted(p)
                    # p_bottom10 = p_sort[0:math.floor(len(p_sort)/10)]
                    # aver = np.mean(p_bottom10)
                    # MIDDLE80の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top80 = p_sort[math.floor(len(p_sort)*1/10):math.floor(len(p_sort)*9/10)]
                    # print(math.floor(len(p_sort)*10/100), math.floor(len(p_sort)*90/100))
                    # aver= np.mean(p_top80)
                    # MIDDLE50の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top50 = p_sort[math.floor(len(p_sort)*25/100):math.floor(len(p_sort)*75/100)]
                    # aver= np.mean(p_top50)
                    # MIDDLE30の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top30 = p_sort[math.floor(len(p_sort)*35/100):math.floor(len(p_sort)*65/100)]
                    # aver= np.mean(p_top30)
                    # MIDDLE10の平均
                    # p_sort = sorted(p, reverse=True)
                    # p_top10 = p_sort[math.floor(len(p_sort)*45/100):math.floor(len(p_sort)*55/100)]
                    # aver= np.mean(p_top10)
                    print(aver)
                    labels = labels.astype(np.uint16)                 
                    labels[a1,a2] = aver

                cv2.imwrite(r'20230901_fiber\label2\7\not_ave\{}.png'.format(wl), labels)

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))