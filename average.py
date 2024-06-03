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
rem_wave =[910,916,922,928,934,940,946,952,958,964,
               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
               1552,1558,1564,1570,1576,1582,1588,1594,1600] 

wave_length = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
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

def nums(first_number, last_number, step=1):
    return range(first_number, last_number + 1, step)                
                
def main():  
            for wl in wave_length: 
            # for wl in del_wave_length:
            # for wl in nums(1,1024):
                print('=====')
                print(wl)

                img1 = cv2.imread(r'20231225_fiber\60\dark\001\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)               
                img2 = cv2.imread(r'20231225_fiber\60\dark\002\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                img3 = cv2.imread(r'20231225_fiber\60\dark\003\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                # img4 = cv2.imread(r'Hyperspectral_imaging_zyushi_03\henkango\white\004\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
                # img5 = cv2.imread(r'Hyperspectral_imaging_zyushi_03\henkango\white\005\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)       

                # print(img1.dtype, img2.dtype, img3.dtype)
                t = 16
                # ave_img = (img1/t+img2/t+img3/t+img4/t+img5/t)/5
                ave_img = (img1/t+img2/t+img3/t)/3
                # ave_img = (img1/t+img2/t)/2
                ave_img = t*ave_img.astype(np.uint16)

                print(img1.dtype, ave_img.dtype)
                print(np.max(img1), np.max(ave_img))
                # ave_img = ave_img.astype(np.uint16)


                cv2.imwrite(r'20231225_fiber\60\dark\ave\{}.png'.format(wl),ave_img)

                    

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))