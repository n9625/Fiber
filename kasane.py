import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from PIL import Image

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
            t = '2'
            # t = '4_6_11_14'
            # t = '4_6_11_14'
            # t = '2_4_6_7_9_11_14'
            # t='2_6_7_9_11_14'
            # s='2_6_7_9_11_14'
            s =t
            # v = '.ver2'
            img = cv2.imread(r'20230901_fiber\label2\{}\top100\910.png'.format(t), flags=cv2.IMREAD_GRAYSCALE) 
            # img = cv2.imread(r'20230901_fiber\label2\2_6_7_9_11_14\top100\910.png',flags=cv2.IMREAD_GRAYSCALE)
            img1 = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
            # nn =  cv2.imread(r'20230901_fiber\NN\test\4_6_11_14\NN_hm4_6_11_14.ver2.png',flags=cv2.IMREAD_COLOR)
            nn =  cv2.imread(r'20230901_fiber\NN\test\{}\NN_hm{}.png'.format(t,t),flags=cv2.IMREAD_COLOR)
            # nn =  cv2.imread(r'20230901_fiber\NN\test\{}\NN_hm{}{}.png'.format(t,t,v),flags=cv2.IMREAD_COLOR)
            # nn = cv2.cvtColor(nn, cv2.COLOR_BGR2GRAY) 
            n3 = nn.copy()
                   
            print(nn)
            a1,a2,a3 = np.where(nn==[255,0,0])
            # a1,a2,a3 = np.where(nn==[255,255,0])
            # a1,a2= np.where(nn==29)
            # print(a1,a2)
            nn[a1,a2,a3] = 0
            threshold = 5
            n2 = nn.copy()
            n2 = cv2.cvtColor(n2, cv2.COLOR_BGR2GRAY) 
            ret, dst = cv2.threshold(n2, threshold, 255, cv2.THRESH_BINARY)
            c1,c2 = np.where(dst==255)
            d1,d2 = np.where(dst!=255) 
            img1[c1,c2] = 0
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)          
            n3[d1,d2] = [0,0,0]
            b1,b2,b3 = np.where(n3!=0)
            # nn = cv2.cvtColor(nn, cv2.COLOR_GRAY2BGR) 
            # b1,b2 = np.where(nn!=0)
            # print(img.shape, nn.shape)
            # b1,b2,b3 = np.where(nn!=[0,255,0])
            # c1,c2,c3 = np.where(nn!=[0,0,255])
            # d1,d2,d3 = np.where(nn!=[255,255,0])
            # e1,e2,e3 = np.where(nn!=[0,255,255])
            # f1,f2,f3 = np.where(nn!=[255,0,255])
            # g1,g2,g3 = np.where(nn!=[255,255,255])
            # h1,h2,h3 = np.where(nn!=[0,0,0])
            img[b1,b2,b3] = 255
            img1[b1,b2,b3] = 255
            # img[b1,b2] = [0,255,0]
            # img[c1,c2] = [0,0,255]
            # img[d1,d2] = [255,255,0]
            # img[e1,e2] = [0,255,255]
            # img[f1,f2] = [255,0,255]
            # img[g1,g2] = [255,255,255]
            # img[h1,h2] = [0,0,0]
            
            imgnn = cv2.addWeighted(img,0.5,n3,0.5,0)
            imgnn = cv2.cvtColor(imgnn, cv2.COLOR_BGR2RGB)


            cv2.imwrite(r'20230901_fiber\NN\test\{}\NN_hm{}_1.png'.format(t,s), n3)
            cv2.imwrite(r'20230901_fiber\NN\test\{}\NN_hm{}_2.png'.format(t,s), img)
            cv2.imwrite(r'20230901_fiber\NN\test\{}\NN_hm{}_3.png'.format(t,s), imgnn)
            cv2.imwrite(r'20230901_fiber\NN\test\{}\NN_hm{}_4.png'.format(t,s), img1)
            # cv2.imwrite(r'20230901_fiber\NN\test\2_6_7_9_11_14.ver5\NN_hm{}_1.png'.format(t), n3)
            # cv2.imwrite(r'20230901_fiber\NN\test\2_6_7_9_11_14.ver5\NN_hm{}_2.png'.format(t), img)
            # cv2.imwrite(r'20230901_fiber\NN\test\2_6_7_9_11_14.ver5\NN_hm{}_3.png'.format(t), imgnn)
            # cv2.imwrite(r'20230901_fiber\NN\test\2_6_7_9_11_14.ver5\NN_hm{}_4.png'.format(t), img1)
"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))