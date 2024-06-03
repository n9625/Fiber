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
# rem_wave =[652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904] 

# wave_length = [652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]
# del_wave_length = wave_length.copy()
# for rem in rem_wave:
#     #print(rem)
#     del_wave_length.remove(rem)
# print(del_wave_length)
# total_l = len(wave_length)
# l = len(wave_length) - len(rem_wave)


def nums(first_number, last_number, step=1):
    return range(first_number, last_number + 1, step)

                
def main(): 
            # v = '2_6_7_9_11_14'
            # v = '4_6_11_14'

            t = 1024
            sh = 1009
            lists = [[] for _ in range(t)] 
            print(lists[1023])

            for n in nums(1,sh):
            # for n in nums(1,1):
                print('=====')
                print(n)
                s = format(n, '04')
                # print(s)
                img = cv2.imread(r'Hyperspectral_imaging_zyushi_06\henkanmae\nori\{}.tif'.format(s), flags=cv2.IMREAD_UNCHANGED)
                print(img.shape, np.max(img), np.min(img))
                # cv2.imwrite(r'kamera2\{}.png'.format(s), img)
                
                for i in range(1024):
                    lists[i].append(img[i,:])
                    # print(lists[i])

            for i in range(1024):
                img = np.array(lists[i])
                print(img, img.shape, np.max(img), np.min(img))
                cv2.imwrite(r'Hyperspectral_imaging_zyushi_06\henkango\nori\{}.png'.format(i+1), img)

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))