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
                
                # n = (wl-910)/6     
                # D = 962-27 
                # d = 915-86
                # t = (D/d-1)*n/115+1
            #ベース画像の読み込み&情報入力
            # img = cv2.imread(r'20230901_fiber\whitecal\check\{}.png'.format(wl), flags=cv2.IMREAD_UNCHANGED)
            img = cv2.imread(r'grad06.png', flags=cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(1280,1024))
            h,w=img.shape[:2]
            print(img.dtype, np.max(img))

            t=0.9          
            cx=935/1024*1024/1280*t
            cy=935/1024*t
            x1=600 #pixel 640
            y1=537 #pixel 512

            #基準位置を確認するためのポイントを描写する
            #必要なければ以降の２行は削除する
            # cv2.circle(img,center=(x1,y1),radius=10,color=255,thickness=-1)
            # cv2.imwrite('img_point.png',img)

            #画像の拡大作業開始
            x2=x1*cx #pixel
            y2=y1*cy #pixel
            size_after=(int(w*cx), int(h*cy))
            resized_img=cv2.resize(img, dsize=size_after)
            deltax=(w/2-x1)-(resized_img.shape[1]/2-x2)
            deltay=(h/2-y1)-(resized_img.shape[0]/2-y2)

            framey=int(h*cy*2)
            framex=int(w*cx*2)
            finalimg=np.zeros((framey,framex),np.uint8)
            # finalimg=np.zeros((framey,framex),np.uint16)
            finalimg[int(-deltay+framey/2-resized_img.shape[0]/2):int(-deltay+framey/2+resized_img.shape[0]/2),
                    int(-deltax+framex/2-resized_img.shape[1]/2):int(-deltax+framex/2+resized_img.shape[1]/2)]=resized_img
            finalimg=finalimg[int(finalimg.shape[0]/2-h/2):int(finalimg.shape[0]/2+h/2),int(finalimg.shape[1]/2-w/2):int(finalimg.shape[1]/2+w/2)]
            
            # 平行移動する値｜画像を100pxずつ並行移動
            dx,dy = -10,-10

            # 平行移動の変換行列を作成
            afin_matrix = np.float32([[1,0,dx],[0,1,dy]])

            # アファイン変換適用
            afin_img = cv2.warpAffine(finalimg,           # 入力画像
                                      afin_matrix,   # 行列
                                      (w,h)  # 解像度
                                      )

            #結果の出力           
            cv2.imwrite(r'20230901_fiber\ganma\{}.png'.format(2001), afin_img)
            print(np.max(afin_img), afin_img.shape, afin_img.dtype)


"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))