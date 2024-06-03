import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler 
import os
import cv2
import glob
import datetime
from sklearn import preprocessing
import scipy.signal
from pathlib import Path
import seaborn as sns
import time
from itertools import product
# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,
#               970]
wavelength = [496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
              610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,
            910,916,922,928,934,940,946,952,958,964,
              976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
              1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
              1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
              1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
              1552,1558,1564,1570,1576,1582,1588,1594,1600]
# wavelength = [1066, 1072, 1090, 1258, 1372, 1378]
# wavelength = [1126,1162,1204,1288,1312,1366,1438,1450,1546]
# wavelength = [1312,1372,1450,1546]

def show(img_arr):
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_arr)
    plt.axis ('off')
    plt.show()
    plt.close()
    return

def SNV (text):
    """SNV処理関数"""
    sc = StandardScaler()
    f = text.T
    sc.fit(f)
    std = sc.transform(f)
    std2 = std.T
    std2= std2[~np.isinf(std2).any(axis=1), :]
    return std2
 
def nanseikyou_tra (text):
    """軟性鏡解析の大元のスペクトル生データの作成"""
    #以下、画像全体でハイパーキューブ (x,y,λデータ) の作成
    array = []
    for wl in wavelength:
        img = cv2.imread(r'{}/{}.png'.format(text,wl),flags=cv2.IMREAD_UNCHANGED)
        array.append(img)
        #print(img.shape)
    #print(array.shape)
    A = np.array(array) #SVM用3次元配列の作成 奥行76×高さ1024×幅1280
    print('A.shape :',A.shape)
    return A

def SG(array):#array:2D行列
    print(array.shape)
    h, w = array.shape
    abs_ave = np.zeros((h, w))
    print(abs_ave.shape)
    for m in range(h):
        abs_ave[m,:] = scipy.signal.savgol_filter(array[m,:], 15, 3, deriv=0)
        #print(n)
    print(abs_ave.shape)
    return abs_ave

def seaborn_plot(raw,name):
    pixel = raw.shape[0]
    a = np.array([[0,0]])
    for index,wl in enumerate(wavelength):
        #print(wl)
        aa = np.full((pixel,1), int(wl))
        bb = raw[:,index].reshape(pixel,1)
        aa = np.append(aa,bb,axis=1)
        a = np.append(a,aa,axis=0)
        #print(a)
    a = a[1:,:]
    columns = ['wavelength', name]
    df = pd.DataFrame(data=a, columns=columns)
    #print(df)
    
    sns.lineplot(x="wavelength", y=name, data=df,errorbar='sd')
    return


def main():

    # f = Path('datas/20230721/spectrum')
    # if not os.path.exists(f) == True:
    #     os.mkdir(f)
    # number1 = os.listdir('datas/20230721/3_scale')
    # number1 = ['7']
    # print(number1)
    
    # for n1 in number1:
        # print(n1)
        # f1 = f / n1
        # if not os.path.exists(f1) == True:
        #     os.mkdir(f1)
        
    th = 950
    h = 1024
    w = 1280

    # n = '2'
    # n = '4'
    # n = '6'
    # n = '7'
    # n = '9'
    # n = '11'
    # n = '14'
    n = 'sasami'
    # t = 'not_ave'
    t = 'top100'
    # t = 'top80'
    # t = 'top50'
    # t = 'top30'
    # t = 'top10'
    # t = 'bottom80'
    # t = 'bottom50'
    # t = 'bottom30'
    # t = 'bottom10'
    # t = 'middle80'
    # t = 'middle50'
    # t = 'middle30'
    # t = 'middle10'
    # t = 'tyuou_ari'
    # t = 'tyuou_nasi'
    # box = '910~1600'

    # cali_mae = (r'20230901_fiber/label2/{}/{}'.format(n,t))
    cali_mae = (r'20231218_fiber_prop\label\all')
    img = nanseikyou_tra(cali_mae)
    print(np.max(img))
    print(np.min(img))

    """load the mask"""
    # mask_ROI = cv2.imread(r'20230901_fiber/mask/{}/2005.png'.format(n),cv2.IMREAD_GRAYSCALE)
    # mask_ROI = cv2.imread(r'20230901_fiber/mask/{}/2004.png'.format(n),cv2.IMREAD_GRAYSCALE)
    mask_ROI = cv2.imread(r'20231218_fiber_prop\ganma\8.png',cv2.IMREAD_GRAYSCALE)
    mask_ROI = np.asarray(mask_ROI)

    plt.imshow(mask_ROI, cmap="gray")
    plt.show()
    # n=0

    """distinguish tumor or normal"""
    h, w = mask_ROI.shape
    ROIData = []

    # merge = cv2.imread(r'datas/20230721/merge/{}/002/1150_1354_1438.png'.format(n1))
    for i, j in product(range(h), range(w)):
        # """
        # if np.any(img[59:126,i,j]<=0):
        #     merge[i,j] = (255,0,0)
        #     continue
            
        # if np.any(img[17:196,i,j]>=65500):
        #     merge[i,j] = (0,255,0)  
        #     continue
        # """
        tmp = (img[:,i,j]) / 65535.0
        tmp = np.where(tmp<=0, 1.0 / 65535.0, tmp)
        if mask_ROI[i,j] > 128:
            ROIData.append(tmp)


    # show(merge)
    # cv2.imwrite('{}/halation.png'.format(f1), merge)


    ROIData = np.asarray(ROIData)
    print(ROIData.shape)

    # folder = str(f)+'/other'
    # sns.set_palette("prism_r")

    """reflection"""
    # cpath = r'20230901_fiber\3Dcube\{}\ref.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\ref.csv'.format(t, n)
    np.savetxt(cpath, np.array(ROIData), fmt='%.10f', delimiter=',')
    # cpath = r'20230901_fiber\3Dcube\{}\ref_ave.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\ref_ave.csv'.format(t, n)
    np.savetxt(cpath, np.average(ROIData,axis=0), fmt='%.10f', delimiter=',')

    seaborn_plot(ROIData,'reflection')
    plt.title("reflection")
    # plt.savefig(r'20230901_fiber\3Dcube\{}\ref_ave.png'.format(n))
    plt.savefig(r'20231218_fiber_prop\3d\ref_ave.png'.format(t, n))
    plt.show()


    """absorvance"""
    ROIData = -np.log10(ROIData)

    # cpath = r'20230901_fiber\3Dcube\{}\abs.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\abs.csv'.format(t, n)
    np.savetxt(cpath, np.array(ROIData), fmt='%.10f', delimiter=',')
    # cpath = r'20230901_fiber\3Dcube\{}\abs_ave.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\abs_ave.csv'.format(t, n)
    np.savetxt(cpath, np.average(ROIData,axis=0), fmt='%.10f', delimiter=',')

    seaborn_plot(ROIData,'absorvance')
    plt.title("absorvance")
    # plt.savefig(r'20230901_fiber\3Dcube\{}\abs_ave.png'.format(n))
    plt.savefig(r'20231218_fiber_prop\3d\abs_ave.png'.format(t, n))
    plt.show()


    """Gavitzky-Golay fillter"""
    ROIData = SG(np.array(ROIData))

    # cpath = r'20230901_fiber\3Dcube\{}\SG.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\SG.csv'.format(t, n)
    np.savetxt(cpath, np.array(ROIData), fmt='%.10f', delimiter=',')
    # cpath = r'20230901_fiber\3Dcube\{}\SG_ave.csv'.format(n)
    cpath = r'20231218_fiber_prop\3d\SG_ave.csv'.format(t, n)
    np.savetxt(cpath, np.average(ROIData,axis=0), fmt='%.10f', delimiter=',')

    seaborn_plot(ROIData,'Savitzky-Galoy')
    plt.title("Savitzky-Galoy")
    # plt.savefig(r'20230901_fiber\3Dcube\{}\SG_ave.png'.format(n))
    plt.savefig(r'20231218_fiber_prop\3d\SG_ave.png'.format(t, n))
    plt.show()

                    
                
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))