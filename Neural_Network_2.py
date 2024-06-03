#Compovision
#Neural Network
#02
#detection検体ごと,他クラス分類


import numpy as np
from sklearn.preprocessing import StandardScaler 
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
import os
import scipy.signal
import time
from keras.callbacks import Callback
from keras.layers import Dense, Input, Flatten
from keras.models import Model
import keras.backend as K
from sklearn.metrics import f1_score
from pathlib import Path


"""1. Setting"""
#p = ['1','2','4','6','7','9','11','14']
p = ['4','6','11','14','other']
name = p

"""remove wavelength"""
#削除したい波長
wave_length = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
              610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
              970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
              1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
              1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
              1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
              1552,1558,1564,1570,1576,1582,1588,1594,1600]

rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
              610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
              790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,
              970]
# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1078,1084,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]
# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1318,1324,1330,1336,1342,1348,1354,1360,1366,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]
# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1132,1138,1144,1150,1156,1168,1174,1180,1186,1192,1198,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1294,1300,1306,1318,1324,1330,1336,1342,1348,1354,1360,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1444,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]


del_wave_length = wave_length.copy()
for rem in rem_wave:
    print(rem)
    del_wave_length.remove(rem)
print(del_wave_length)
total_l = len(wave_length)
l = len(wave_length) - len(rem_wave)
"""2. function"""

def SNV (text):
    """SNV処理するよ"""
    sc = StandardScaler()
    f = text.T
    sc.fit(f)
    std = sc.transform(f)
    std2 = std.T
    std2 = std2[~np.isinf(std2).any(axis=1), :] #欠損値の除外
    return std2


def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
        
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


def pseudo_jet(x):
    x=[x]
    
    jet_color = 256*x
    rgb = plt.cm.jet(jet_color)
    r, g, b = rgb[0:1,0], rgb[0:1,1], rgb[0:1,2]
    
    #color = [int(255 * b[0]), int(255* g[0]), int(255 * r[0])]
    color = [int(255 * r[0]), int(255* g[0]), int(255 * b[0])]
    
    return color


def nanseikyou_tra (text):
    '''軟性鏡解析の大元のスペクトル生データの作成'''
    #以下、画像全体でハイパーキューブ (x,y,λデータ) の作成
    array = []
    for wl in del_wave_length:
        img = cv2.imread(r'{}/{}.png'.format(text,wl), flags=cv2.IMREAD_UNCHANGED)
        array.append(img)
    A = np.array(array) #SVM用3次元配列の作成
    B = A[:,0].T #numpyの0行目にアクセス、奥行方向(波長情報)抽出

    for j in range(1,1024):
        C = A[:,j].T
        B = np.concatenate((B,C), axis = 0)
        #B= cp.asarray(B)
        
    return B

# def remove_wavelength(arr,rem_wave,wave_length):
#     """波長を削除"""      
#     rem_list = []
#     for r in rem_wave:
#         rem_no = wave_length.index(r)
#         rem_list.append(rem_no)        
    
#     del_arr = np.delete(arr, rem_list, 1)
#     return del_arr

def SG2(array):#array:2D行列
    print(array.shape)
    h, w = array.shape
    abs_ave = np.zeros((h, w))
    print(abs_ave.shape)
    for m in range(h):
        abs_ave[m,:] = scipy.signal.savgol_filter(array[m,:], 15, 3, deriv=0)
        #print(n)
    print(abs_ave.shape)
    return abs_ave

def show(img_arr):
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_arr)
    plt.axis ('off')
    plt.show()
    plt.close()
    return


def main():
    seido_val = []
    seido_tumor = []
    seido_normal = []
    seido_zentai = []
    TP_count = []
    FP_count = []
    TN_count = []
    FN_count = []
    
    """detection"""
    # f1 = Path('datas/20230721/NN')
    # if not os.path.exists(f1) == True:
    #     os.mkdir(f1)

    print('==========')    
    # name = '490-1000_3layer_300epo_10000pix_batch32_ES_C4-6-11-14-other'
    # result_path = str(f1) + '/result_' + name# <-- Input
    # if not os.path.exists(result_path) == True:
    #     os.mkdir(result_path)
    # gakushuki_path = str(f1) + '/gakushuki_' + name # <-- Input
    # m_path = gakushuki_path+'/'

    # gakushuki_path = '20230901_fiber/NN/learning'
    # m_path = gakushuki_path+'/'

    # t = 'top100'
    # t = 'top80'
    # t = 'top50'
    # t = 'top30'
    # t = 'top10'
    # t = 'middle80'
    # t = 'middle50'
    # t = 'middle30'
    # t = 'middle10'
    # t = 'bottom80'
    # t = 'bottom50'
    # t = 'bottom30'
    # t = 'bottom10'
    # t = 'not_ave'
    # t = 'tyuou_ari'
    # t = 'tyuou_nasi'
    t = 'tyuou_r2'
    # t = 'tyuou_r1'
    # t = 'tyuou_1tenn'
    # lab = '2'
    # lab = '4_6_11_14'
    lab = '2_6_7_9_11_14'
    # lab = '2_4_6_7_9_11_14'
    # n = '_new'
    # ver = '.ver1'
    # ver = '.ver2'
    # ver = '.ver3'
    # ver = '.ver4'
    # ver = '.ver5'
    # ver = '.ver6'
    # ver = '.ver7'
    # ver = '.verSENTEI3.2'
    # SG = '_SG5_3'
    # SG = '_SG10_3'
    SG = '_SG15_3'
    # c = '_22615'
    
    """↓If you don't check the Precision or F-measure"""
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}\best_model.h5'.format(lab))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}{}\best_model.h5'.format(lab,ver))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}{}\best_model.h5'.format(lab,SG))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}{}{}\best_model.h5'.format(lab,SG,ver))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}{}{}\best_model.h5'.format(lab,SG,c))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\top100\{}{}{}{}\best_model.h5'.format(lab,SG,ver,c))
    # loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\{}\{}\best_model.h5'.format(t,lab))
    loaded_model = tf.keras.models.load_model(r'20230901_fiber\NN\learning\{}\{}{}\best_model.h5'.format(t,lab,SG))
    
    # lab = '4_6_11_14'
    # lab = '2_6_7_9_11_14'

    """processing"""       
    cali_mae = (r'20230901_fiber\label2\{}\{}'.format(lab,t))
    test_raw = nanseikyou_tra(cali_mae)
    test = -np.log10(test_raw/(65535))
    test[np.isnan(test)] = 0
    test[np.isinf(test)] = 0
    # test = SG2(test)
    test_snv =test
    # test_snv = remove_wavelength(test,rem_wave,wave_length) #使わない波長を削除
    #test_snv = SNV(test_snv)
    print(test_snv)
    print(test_snv.shape) #[1024*1280, 波長数]

    """show the figure"""
    plt.plot(del_wave_length, np.average(test_snv, axis = 0))
    plt.show()

    """detection"""
    pred_test = loaded_model.predict(test_snv) #[1024*1280, class数]

    pred_test = np.argmax(pred_test, axis=1)
    SVM_result = np.reshape(pred_test, (1024,1280))

    SVM_result2 = SVM_result

    x = 1280
    y = 1024

    heat = cv2.imread(r'20230901_fiber\label2\{}\{}\910.png'.format(lab,t),flags=cv2.IMREAD_GRAYSCALE)
    heat = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
    show(heat)

    for row in range (y):
        for col in range (x):
            # if SVM_result2[row][col]==0:
            #     heat[row][col] = (255,0,0)
            # if SVM_result2[row][col]==1:
            #     heat[row][col] = (0,255,0)
            # if SVM_result2[row][col]==2:
            #     heat[row][col] = (255,255,0)
            # if SVM_result2[row][col]==3:
            #     heat[row][col] = (0,0,255)                             
            # if SVM_result2[row][col]==4:
            #     # continue
            #     heat[row][col] = (0,255,255)                           
            # if SVM_result2[row][col]==5:
            #     heat[row][col] = (255,0,255)                              
            # if SVM_result2[row][col]==6:
            #     # continue
            #     heat[row][col] = (255,255,255)
            # if SVM_result2[row][col]==7:
            #     heat[row][col] = (0,0,0)
            #     # continue

            # 6zyushi
            if SVM_result2[row][col]==0:
                heat[row][col] = (0,0,0)
            if SVM_result2[row][col]==1:
                heat[row][col] = (255,0,0)
            if SVM_result2[row][col]==2:
                heat[row][col] = (0,255,0)
            if SVM_result2[row][col]==3:
                heat[row][col] = (0,0,255)                             
            if SVM_result2[row][col]==4:
                heat[row][col] = (255,255,0)                           
            if SVM_result2[row][col]==5:
                heat[row][col] = (0,255,255)                              
            if SVM_result2[row][col]==6:
                heat[row][col] = (255,0,255)

            # 4zyushi
            # if SVM_result2[row][col]==0:
            #     heat[row][col] = (0,0,0)
            # if SVM_result2[row][col]==1:
            #     heat[row][col] = (55,82,177)
            # if SVM_result2[row][col]==2:
            #     heat[row][col] = (0,255,0)                          
            # if SVM_result2[row][col]==3:
            #     heat[row][col] = (0,255,255)                              
            # if SVM_result2[row][col]==4:
            #     heat[row][col] = (255,0,255)
            

    show(heat)
    cv2.imwrite(r'20230901_fiber\NN\test\{}\NN_hm{}.png'.format(t,lab),heat) 

        
            
"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))
