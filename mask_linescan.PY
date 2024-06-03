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
            img = cv2.imread(r'Hyperspectral_imaging_zyushi_03\henkango\whitecal\2\{}.png'.format(910), flags=cv2.IMREAD_GRAYSCALE)  #2
            # img = cv2.imread(r'20230901_fiber\label2\4\top100\{}.png'.format(1426), flags=cv2.IMREAD_GRAYSCALE)  #4
            # img = cv2.imread(r'20230901_fiber\label2\6\top100\{}.png'.format(910), flags=cv2.IMREAD_GRAYSCALE)  #6
            # img = cv2.imread(r'20230901_fiber\label2\7\top100\{}.png'.format(910), flags=cv2.IMREAD_GRAYSCALE)  #7
            # img = cv2.imread(r'20230901_fiber\label2\9\top100\{}.png'.format(1534), flags=cv2.IMREAD_GRAYSCALE)  #9
            # img = cv2.imread(r'20230901_fiber\label2\11\top100\{}.png'.format(1414), flags=cv2.IMREAD_GRAYSCALE)  #11
            # img = cv2.imread(r'20230901_fiber\label2\14\top100\{}.png'.format(1210), flags=cv2.IMREAD_GRAYSCALE)  #14
            zero = img.copy()
            zero[:,:] = 0

            plt.imshow(img)
            plt.show()

            #zyushi
            #2
            # mask_poly1 = np.array([[627,421],[630,551],[778,551],[767,418]])
            # mask_poly2 = np.array([[519,613],[520,748],[657,748],[652,612]])             
            # mask_poly3 = np.array([[736,618],[732,754],[859,747],[866,628]])
            #4
            # mask_poly1 = np.array([[586,454],[583,591],[729,585],[731,453]])
            # mask_poly2 = np.array([[485,667],[488,789],[629,785],[612,658]])             
            # mask_poly3 = np.array([[686,643],[693,774],[829,776],[831,644]])
            6
            # mask_poly1 = np.array([[573,450],[567,566],[713,562],[710,446]])
            # mask_poly2 = np.array([[478,647],[471,767],[594,767],[601,643]])             
            # mask_poly3 = np.array([[667,632],[663,750],[779,756],[793,629]])
            # 7
            # mask_poly1 = np.array([[597,455],[591,589],[733,593],[739,458]])
            # mask_poly2 = np.array([[513,644],[521,774],[644,767],[647,639]])             
            # mask_poly3 = np.array([[703,656],[701,774],[851,779],[846,652]])
            #9
            # mask_poly1 = np.array([[621,489],[604,632],[760,630],[768,493]])
            # mask_poly2 = np.array([[520,694],[505,821],[643,827],[652,706]])             
            # mask_poly3 = np.array([[710,686],[703,833],[852,832],[839,685]])
            11
            # mask_poly1 = np.array([[607,465],[607,591],[749,584],[755,465]])
            # mask_poly2 = np.array([[509,643],[512,768],[650,771],[647,645]])             
            # mask_poly3 = np.array([[712,636],[714,762],[850,748],[847,623]])
            #14
            # mask_poly1 = np.array([[560,437],[560,554],[696,551],[705,438]])
            # mask_poly2 = np.array([[458,619],[440,736],[594,753],[608,628]])             
            # mask_poly3 = np.array([[661,617],[660,741],[814,736],[809,610]])

            #mawari
            #2
            mask_poly1 = np.array([[595,396],[612,574],[803,573],[789,396]])
            mask_poly2 = np.array([[503,601],[502,766],[684,772],[674,594]])             
            mask_poly3 = np.array([[718,598],[716,770],[879,765],[891,606]])
            # #4
            #6
            #7
            #9
            #11
            #14



            img_mask = cv2.fillPoly(img, [mask_poly1], 255)
            img_mask = cv2.fillPoly(img_mask, [mask_poly2], 255)
            img_mask = cv2.fillPoly(img_mask, [mask_poly3], 255)
            mask = cv2.fillPoly(zero, [mask_poly1], 255)
            mask = cv2.fillPoly(zero, [mask_poly2], 255)
            mask = cv2.fillPoly(mask, [mask_poly3], 255)
         
           
            cv2.imwrite(r'Hyperspectral_imaging_zyushi_03\henkango\mask\0\{}.png'.format(1), img_mask)
            cv2.imwrite(r'Hyperspectral_imaging_zyushi_03\henkango\mask\0\{}.png'.format(2), mask)

            a1,a2 = np.where(mask==255)
            b1,b2 = np.where(mask!=255)
            h, w = img.shape[:2]
            mask1 = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask1, center=(669,519), radius=400 , color=255, thickness=-1)
            mask[mask1==255] = 255     
            mask[a1,a2] = 0
            # mask[b1,b2] = 255
            

            cv2.imwrite(r'Hyperspectral_imaging_zyushi_03\henkango\mask\0\{}.png'.format(3), mask)
            

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))