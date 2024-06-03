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


def main():   
            # for wl in del_wave_length:
            #     print('=====')
            #     print(wl)
            mask = cv2.imread(r'20230901_fiber\mask\2100.png', flags=cv2.IMREAD_UNCHANGED)
            img = cv2.imread(r'20230901_fiber\label2\4\top100\{}.png'.format(1426), flags=cv2.IMREAD_GRAYSCALE)  #4
            # 連結成分のラベリングを行う。
            n_labels, labels = cv2.connectedComponents(mask)

            # # ラベル数
            # print("number of labels:", n_labels)
            # fig, ax = plt.subplots(figsize=(7, 7))
            # ax.imshow(labels)
            # plt.show()

            # num = list(range(n_labels))
            # print(num)
            num_rem = [
                       166, 313, 380, 398, 405, 413, 414, 419, 428, 442, 443, 448, 452, 457, 458, 459, 467, 474, 476, 477, 
                       
                       488, 491, 492, 493, 500, 501, 504, 511, 514,
                       515, 519, 523, 529, 535, 536, 537, 544, 545, 546, 555, 556, 559, 560, 568, 569, 573, 580, 585, 588, 589, 595, 598, 599, 602, 
                       
                       608, 612, 622, 623, 624, 625, 
                       631, 636, 637, 646, 647, 
                       651, 652, 657, 663, 669, 670, 673, 679, 680, 683, 692, 693, 698, 
                       705, 708, 709, 713, 714, 722, 732, 745, 747, 786, 787, 790, 800,
                       749, 755, 756, 757, 767, 768, 773, 779, 792, 802, 809, 810, 824, 825, 834, 838, 839, 857, 858, 864, 876, 882, 891, 907, 910, 931, 932, 949, 
                       1193, 1211, 

                       1216, 1219, 1233, 1234, 1247, 1260, 1263, 1265, 1278, 1286, 1287, 1288, 1291, 1295, 1296, 1301, 1309, 1313, 1316, 1321, 1322, 1326, 1328, 
                       1332, 1339, 1340, 1341, 1344, 1347, 1351, 1355, 1356, 1358, 1359, 1362, 1363, 1368, 1369, 1374, 1376, 1379, 1380, 1382, 1387, 1391, 1396, 
                       1397, 1398, 1399, 1401, 1402, 1403, 1408, 1414, 1418, 1419, 1420, 1422, 1424, 1432, 1436, 1437, 1443, 1448, 1450, 1456, 1460, 1471, 1473, 
                       1474, 1481, 1484, 1490, 1494, 1504, 1506, 1508, 1510, 1520, 1531, 1548, 1551, 1555, 1563, 1565, 1573, 1596, 1602, 1606, 1608, 1637, 1641, 
                       1647, 1650, 1653, 1657, 1660, 1670, 1671, 1677, 1679, 1693, 1720, 1812
                       ] #4
            # for n in num_rem:
            #     num.remove(n)
            # print(num)
            img2 = img.copy()
            mask3 = mask.copy()
            mask3[:,:] = 0
            # print(mask3)
            img4 = img.copy()
            img4[:,:] = 0
            img5 = img.copy()
            

            for i in num_rem: 
                    a1,a2 = np.where(labels==i)
                    img2[a1,a2] = 0 #2
                    mask3[a1,a2] = 255 #3
                    img4[a1,a2] = img[a1,a2]
                    img5[a1,a2] = 255 #5

            cv2.imwrite(r'20230901_fiber\mask\4\{}.png'.format(920), img)
            cv2.imwrite(r'20230901_fiber\mask\4\{}.png'.format(921), img2)
            cv2.imwrite(r'20230901_fiber\mask\4\{}.png'.format(922), mask3)
            cv2.imwrite(r'20230901_fiber\mask\4\{}.png'.format(923), img4)
            cv2.imwrite(r'20230901_fiber\mask\4\{}.png'.format(924), img5)


"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))