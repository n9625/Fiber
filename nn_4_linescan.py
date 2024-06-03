#Compovision
#Neural Network
#01
#learning
#多クラス分類

# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import os
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
import sys 
import time
from pathlib import Path
import scipy.signal

"""1. Setting"""
#p = ['1','2','4','6','7','9','11','14']
#p = ['2','6','7','9','11','14']
# p = ['4','6','11','14','other']
# p = ['0','2','4','6','7','9','11','14']
p = ['0','4','6','11','14']
name = p

mag = 1  #腫瘍の何倍の画素をとるか

fold_size = 3 #crossvalidation

# num = 'not_ave'
# num = 'top100'
# num = 'top80'
# num = 'top50'
# num = 'top30'
# num = 'top10'
# num = 'bottom80'
# num = 'bottom50'
# num = 'bottom30'
num = 'bottom10'
# num = 'middle80'
# num = 'middle50'
# num = 'middle30'
# num = 'middle10'


"""remove wavelength"""
#削除したい波長
# wave_length = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1312,1318,1324,1330,1336,1342,1348,1354,1360,1366,1372,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1450,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,1546,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]

# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,
#               970]

# rem_wave = [430,436,442,448,454,460,466,472,478,484,490,496,502,508,514,520,526,532,538,544,550,556,562,568,574,580,586,592,598,604,
#               610,616,622,628,634,640,646,652,658,664,670,676,682,688,694,700,706,712,718,724,730,736,742,748,754,760,766,772,778,784,
#               790,796,802,808,814,820,826,832,838,844,850,856,862,868,874,880,886,892,898,904,910,916,922,928,934,940,946,952,958,964,
#               970,976,982,988,994,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,1060,1066,1072,1078,1084,1090,1096,1102,1108,1114,
#               1120,1126,1132,1138,1144,1150,1156,1162,1168,1174,1180,1186,1192,1198,1204,1210,1216,1222,1228,1234,1240,1246,1252,1258,
#               1264,1270,1276,1282,1288,1294,1300,1306,1318,1324,1330,1336,1342,1348,1354,1360,1366,1378,1384,1390,1396,1402,
#               1408,1414,1420,1426,1432,1438,1444,1456,1462,1468,1474,1480,1486,1492,1498,1504,1510,1516,1522,1528,1534,1540,
#               1552,1558,1564,1570,1576,1582,1588,1594,1600]

# del_wave_length = wave_length.copy()
# for rem in rem_wave:
#     print(rem)
#     del_wave_length.remove(rem)
# print(del_wave_length)
# total_l = len(wave_length)
# l = len(wave_length) - len(rem_wave)

# ROOT = 'datas/20230721'

def nums(first_number, last_number, step=1):
    return range(first_number, last_number + 1, step) 

wavelength =[]
# for wl in nums(1,1024):
#     wavelength.append(wl)
# print(wavelength)   

wavelength.append(1)
for wl in nums(1,146):
    wl = wl*7+1
    wavelength.append(wl)
print(wavelength)  

l = len(wavelength)

"""2. function"""
'''SNV'''
def SNV (text):
    """SNV処理するよ"""
    sc = StandardScaler()
    f = text.T
    sc.fit(f)
    std = sc.transform(f)
    std2 = std.T
    std2 = std2[~np.isinf(std2).any(axis=1), :] #欠損値の除外
    return std2

"""remove wavelength"""
# def remove_wavelength(arr,rem_wave,wave_length):
#     """波長を削除"""      
#     rem_list = []
#     for r in rem_wave:
#         rem_no = wave_length.index(r)
#         rem_list.append(rem_no)        
    
#     del_arr = np.delete(arr, rem_list, 1)
    
#     return del_arr

"""Savitzky-Golay"""
def SG(array):#array:2D行列
    print('SG shape :',array.shape)
    h, w = array.shape
    abs_ave = np.zeros((h, w))
    for m in range(h):
        abs_ave[m,:] = scipy.signal.savgol_filter(array[m,:], 15, 3, deriv=0)
        #print(n)
    return abs_ave

"""make random datasets for training"""
def CV_training (p):
    training_0 = np.reshape(np.arange(l), (1,l))
    training_1 = np.reshape(np.arange(l), (1,l))
    training_2 = np.reshape(np.arange(l), (1,l))
    training_3 = np.reshape(np.arange(l), (1,l))
    training_4 = np.reshape(np.arange(l), (1,l))
    training_5 = np.reshape(np.arange(l), (1,l))
    training_6 = np.reshape(np.arange(l), (1,l))
    training_7 = np.reshape(np.arange(l), (1,l))

        
    a = np.loadtxt(r'Hyperspectral_imaging_zyushi_03/henkango/3d/{}/abs.csv'.format(p[0]), delimiter=",")[:,:]
    # a = remove_wavelength(a,rem_wave,wave_length)
    a = SG(a)
    np.random.shuffle(a) #教師データのランダム抽出
    aa = a[:10000,:] #抽出したい数を入力10000, 22615
    training_0=np.append(training_0, aa, axis=0)

    mag_p = int(mag * aa.shape[0]) #tumorの何倍いれるか

    b = np.loadtxt(r'Hyperspectral_imaging_zyushi_03/henkango/3d/{}/abs.csv'.format(p[1]), delimiter=",")[:,:]
    # b = remove_wavelength(b,rem_wave,wave_length)       
    b = SG(b)     
    np.random.shuffle(b) #教師データのランダム抽出
    bb = b[0:mag_p,:] #抽出したい数を入力
    training_1=np.append(training_1, bb, axis=0)

    c = np.loadtxt(r'Hyperspectral_imaging_zyushi_03/henkango/3d/{}/abs.csv'.format(p[2]), delimiter=",")[:,:]
    # c = remove_wavelength(c,rem_wave,wave_length)    
    c = SG(c)        
    np.random.shuffle(c) #教師データのランダム抽出
    cc = c[0:mag_p,:] #抽出したい数を入力
    training_2=np.append(training_2, cc, axis=0)

    d = np.loadtxt(r'Hyperspectral_imaging_zyushi_03/henkango/3d/{}/abs.csv'.format(p[3]), delimiter=",")[:,:]
    # d = remove_wavelength(d,rem_wave,wave_length)  
    d = SG(d)          
    np.random.shuffle(d) #教師データのランダム抽出
    dd = d[0:mag_p,:] #抽出したい数を入力
    training_3=np.append(training_3, dd, axis=0)
    
    
    e = np.loadtxt(r'Hyperspectral_imaging_zyushi_03/henkango/3d/{}/abs.csv'.format(p[4]), delimiter=",")[:,:]
    # e = remove_wavelength(e,rem_wave,wave_length)
    e = SG(e)
    np.random.shuffle(e) #教師データのランダム抽出
    ee = e[0:mag_p,:] #抽出したい数を入力
    training_4=np.append(training_4, ee, axis=0)

    
    # f = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(p[5]), delimiter=",")[:,:]
    # # f = remove_wavelength(f,rem_wave,wave_length)            
    # np.random.shuffle(f) #教師データのランダム抽出
    # ff = f[0:mag_p,:] #抽出したい数を入力
    # training_5=np.append(training_5, ff, axis=0)

    
    
    # g = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(p[6]), delimiter=",")[:,:]
    # # g = remove_wavelength(g,rem_wave,wave_length)            
    # np.random.shuffle(g) #教師データのランダム抽出
    # gg = g[0:mag_p,:] #抽出したい数を入力
    # training_6=np.append(training_6, gg, axis=0)
    
    
    # h = np.loadtxt(r'20230901_fiber\3Dcube\{}\abs.csv'.format(p[7]), delimiter=",")[:,:]
    # # h = remove_wavelength(h,rem_wave,wave_length)            
    # np.random.shuffle(h) #教師データのランダム抽出
    # hh = h[0:mag_p,:] #抽出したい数を入力
    # training_7=np.append(training_7, hh, axis=0)
    
                    
    tra0 = training_0[1:,:] #余計な部分の削除
    tra1 = training_1[1:,:] #余計な部分の削除
    tra2 = training_2[1:,:] #余計な部分の削除
    tra3 = training_3[1:,:] #余計な部分の削除
    tra4 = training_4[1:,:] #余計な部分の削除
    # tra5 = training_5[1:,:] #余計な部分の削除
    # tra6 = training_6[1:,:] #余計な部分の削除
    # tra7 = training_7[1:,:] #余計な部分の削除
    
    return tra0, tra1, tra2, tra3, tra4#, tra5, tra6, tra7

"""make Neural Network model"""
def gakusyuuki ():
    #以下学習データの作成
    model = keras.Sequential()
    #各種活性化関数の決定（要検討）
    model.add(keras.layers.Dense(256,activation='sigmoid', use_bias=True, input_dim=l))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(512, use_bias=True, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1024, use_bias=True, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.Dense(2048, use_bias=True, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.Dense(1024, use_bias=True, activation='relu'))
    #####
    #model.add(keras.layers.Dropout(0.3))
    #model.add(keras.layers.Dense(2048, use_bias=True, activation='relu'))
    #####
    
    # kokokara
# =============================================================================
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.Dense(512, use_bias=True, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.Dense(256, use_bias=True, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.Dense(128, use_bias=True, activation='relu'))
# =============================================================================
    # kokomade
    
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(5,use_bias=True, activation='softmax'))
    #モデル評価方法の決定(要検討）
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])

    return model 


def main():
    # f1 = Path('{}/NN'.format(ROOT))
    # if not os.path.exists(f1) == True:
    #     os.mkdir(f1)

    # gakushuki_path = str(f1) + '/gakushuki_' + '490-1000_3layer_300epo_10000pix_batch32_ES_C4-6-11-14-other' # <-- Input
    # if not os.path.exists(gakushuki_path) == True:
    #     os.mkdir(gakushuki_path)

    """make the training data"""
    print('テスト')
    tra0, tra1, tra2, tra3, tra4 =  CV_training(p) ####変える必要あり
    # , tra5, tra6, tra7

    sh_row, sh_col = tra0.shape
    print(sh_row)
    """SNV"""
    #tra0 = SNV(tra0)
    tr0 = np.array_split(tra0, fold_size)
    #tra1 = SNV(tra1)
    tr1 = np.array_split(tra1, fold_size)
    if len(name)>=3:
        #tra2 = SNV(tra2)
        tr2 = np.array_split(tra2, fold_size)
    if len(name)>=4:  
        #tra3 = SNV(tra3)
        tr3 = np.array_split(tra3, fold_size)
             
    if len(name)>=5:
        #tra4 = SNV(tra4)
        tr4 = np.array_split(tra4, fold_size)
        
        
    # if len(name)>=6:  
    #     #tra5 = SNV(tra5)
    #     tr5 = np.array_split(tra5, fold_size)
    
    
    # if len(name)>=7:
    #     #tra6 = SNV(tra6)
    #     tr6 = np.array_split(tra6, fold_size)
        
    # if len(name)>=8:  
    #     # tra7 = SNV(tra7)
    #     tr7 = np.array_split(tra7, fold_size)
        

    """learning"""
    loss2=[]
    maxf2=[]
    pre2=[]
    vec = np.arange(fold_size)
    for i in range(0,fold_size):
        model = gakusyuuki()
        # m_path =gakushuki_path + '/'

        """training data"""
        vec_tra = i!=vec
        tra_No = vec[np.array(vec_tra)]
        t0_tra = np.arange(l).reshape(1,l)
        t1_tra = np.arange(l).reshape(1,l)
        if len(name)>=3:
            t2_tra = np.arange(l).reshape(1,l)
        if len(name)>=4:
            t3_tra = np.arange(l).reshape(1,l)
                     
        if len(name)>=5:
            t4_tra = np.arange(l).reshape(1,l)
            
            
        # if len(name)>=6:
        #     t5_tra = np.arange(l).reshape(1,l)
        
        # if len(name)>=7:
        #     t6_tra = np.arange(l).reshape(1,l)
            
        # if len(name)>=8:
        #     t7_tra = np.arange(l).reshape(1,l)
            

        for m in range(fold_size-1):    
            t0_tra2 = tr0[tra_No[m]]
            t1_tra2 = tr1[tra_No[m]]  
            if len(name)>=3:
                t2_tra2 = tr2[tra_No[m]]  
            if len(name)>=4:
                t3_tra2 = tr3[tra_No[m]]  
                
            if len(name)>=5:
                t4_tra2 = tr4[tra_No[m]]  
                
                
            # if len(name)>=6:
            #     t5_tra2 = tr5[tra_No[m]]  
            
            # if len(name)>=7:
            #     t6_tra2 = tr6[tra_No[m]]  
                
            # if len(name)>=8:
            #     t7_tra2 = tr7[tra_No[m]]  
                

            t0_tra = np.append(t0_tra, t0_tra2, axis=0)
            t1_tra = np.append(t1_tra, t1_tra2, axis=0)
            if len(name)>=3:
                t2_tra = np.append(t2_tra, t2_tra2, axis=0)
            if len(name)>=4:
                t3_tra = np.append(t3_tra, t3_tra2, axis=0)
                
            if len(name)>=5:
                t4_tra = np.append(t4_tra, t4_tra2, axis=0)
                
                
            # if len(name)>=6:
            #     t5_tra = np.append(t5_tra, t5_tra2, axis=0)
            
            # if len(name)>=7:
            #     t6_tra = np.append(t6_tra, t6_tra2, axis=0)
                
            # if len(name)>=8:
            #     t7_tra = np.append(t7_tra, t7_tra2, axis=0)
                


        t0_tra = t0_tra[1:,:]
        t1_tra = t1_tra[1:,:]
        if len(name)>=3:
            t2_tra = t2_tra[1:,:]
        if len(name)>=4:
            t3_tra = t3_tra[1:,:]
            
        if len(name)>=5:
            t4_tra = t4_tra[1:,:]
            
            
        # if len(name)>=6:
        #     t5_tra = t5_tra[1:,:]
        
        # if len(name)>=7:
        #     t6_tra = t6_tra[1:,:]
            
        # if len(name)>=8:
        #     t7_tra = t7_tra[1:,:]
            

        """labeling"""
        t0_tra_y = np.zeros((t0_tra.shape[0],len(name)))
        t0_tra_y[:,0] = 1
        t1_tra_y = np.zeros((t1_tra.shape[0],len(name)))
        t1_tra_y[:,1] = 1
        if len(name)>=3:
            t2_tra_y = np.zeros((t2_tra.shape[0],len(name)))
            t2_tra_y[:,2] = 1
        if len(name)>=4:
            t3_tra_y = np.zeros((t3_tra.shape[0],len(name)))
            t3_tra_y[:,3] = 1
            
        if len(name)>=5:
            t4_tra_y = np.zeros((t4_tra.shape[0],len(name)))
            t4_tra_y[:,4] = 1
            
            
        # if len(name)>=6:
        #     t5_tra_y = np.zeros((t5_tra.shape[0],len(name)))
        #     t5_tra_y[:,5] = 1
        
        # if len(name)>=7:
        #     t6_tra_y = np.zeros((t6_tra.shape[0],len(name)))
        #     t6_tra_y[:,6] = 1
            
        # if len(name)>=8:
        #     t7_tra_y = np.zeros((t7_tra.shape[0],len(name)))
        #     t7_tra_y[:,7] = 1
            

        training_data = np.r_[t0_tra,t1_tra,t2_tra,t3_tra,t4_tra]####変える必要あり
        # ,t5_tra,t6_tra,t7_tra
        training_data_y = np.r_[t0_tra_y,t1_tra_y,t2_tra_y,t3_tra_y,t4_tra_y]####変える必要あり
        # ,t5_tra_y,t6_tra_y,t7_tra_y

        print(training_data_y)
        print(training_data.shape)
        print(training_data_y.shape)


        """test data"""
        vec_val = i==vec
        val_No = vec[np.array(vec_val)]
        t0_val = tr0[val_No[0]]
        t1_val = tr1[val_No[0]]
        if len(name)>=3:
            t2_val = tr2[val_No[0]]
        if len(name)>=4:
            t3_val = tr3[val_No[0]]
            
        if len(name)>=5:
            t4_val = tr4[val_No[0]]
            
           
        # if len(name)>=6:
        #     t5_val = tr5[val_No[0]]
            
        # if len(name)>=7:
        #     t6_val = tr6[val_No[0]]
            
        # if len(name)>=8:
        #     t7_val = tr7[val_No[0]]
            


        """labeling"""
        t0_val_y = np.zeros((t0_val.shape[0],len(name)))
        t0_val_y[:,0] = 1
        t1_val_y = np.zeros((t1_val.shape[0],len(name)))
        t1_val_y[:,1] = 1
        if len(name)>=3:
            t2_val_y = np.zeros((t2_val.shape[0],len(name)))
            t2_val_y[:,2] = 1
        if len(name)>=4:
            t3_val_y = np.zeros((t3_val.shape[0],len(name)))
            t3_val_y[:,3] = 1
            
        if len(name)>=5:
            t4_val_y = np.zeros((t4_val.shape[0],len(name)))
            t4_val_y[:,4] = 1
            
            
        # if len(name)>=6:
        #     t5_val_y = np.zeros((t5_val.shape[0],len(name)))
        #     t5_val_y[:,5] = 1
            
        # if len(name)>=7:
        #     t6_val_y = np.zeros((t6_val.shape[0],len(name)))
        #     t6_val_y[:,6] = 1
            
        # if len(name)>=8:
        #     t7_val_y = np.zeros((t7_val.shape[0],len(name)))
        #     t7_val_y[:,7] = 1
            


        validation_data = np.r_[t0_val,t1_val,t2_val,t3_val,t4_val]####変える必要あり
        # ,t5_val,t6_val,t7_val
        validation_data_y = np.r_[t0_val_y,t1_val_y,t2_val_y,t3_val_y,t4_val_y]####変える必要あり
        # ,t5_val_y,t6_val_y,t7_val_y

        print(validation_data_y)
        print(validation_data.shape)
        print(validation_data_y.shape)

        # gakushuki_path = '20230901_fiber/NN/learning'
        # m_path = gakushuki_path +'/'
        
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,mode='auto'),
            tf.keras.callbacks.ModelCheckpoint(filepath= r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/{}otameshi.h5'.format(str(i)),save_best_only=True),
        ]

        """learning"""
        history = model.fit(training_data,
                            training_data_y,
                            epochs=300,
                            batch_size=32,
                            validation_data=(validation_data, validation_data_y),
                            callbacks=my_callbacks,
                            verbose=1)



        """show laerning curve"""
        metrics = ['loss', 'accuracy']  # 使用する評価関数を指定
        plt.figure(figsize=(15, 5))  # グラフを表示するスペースを用意
        # plt.title(str(i))
        for m in range(len(metrics)):
            metric = metrics[m]
            plt.subplot(1, 2, m+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
            plt.title(metric)  # グラフのタイトルを表示
            plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
            plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す

            plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
            plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
            plt.xlabel('Epochs')
            plt.ylabel(metrics[m])
            plt.legend()  # ラベルの表示

            pd.DataFrame(plt_train).to_csv(r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/{}{}_gakushuritsu_train.csv'.format(str(i), metrics[m]))
            pd.DataFrame(plt_test).to_csv(r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/{}{}_gakushuritsu_test.csv'.format(str(i), metrics[m]))
        plt.savefig(r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/{}_gakushuritsu.png'.format(str(i)))
        plt.show()  # グラフの表示

        """"save the best model"""
        model.save(filepath=r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/classification_model.h5', save_format='h5')
        history_dict = history.history
        acc = np.array(history_dict['accuracy'])
        val_acc = np.array(history_dict['val_accuracy'])
        loss = np.array(history_dict['loss'])
        val_loss = np.array(history_dict['val_loss'])
        history_dict2 = history_dict.update(history_dict)
        min_loss = np.min(val_loss)
        loss2 = np.append(loss2,min_loss)


    best_model_number = np.argmin(loss2)
    print(best_model_number)
    os.rename(r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/{}otameshi.h5'.format(str(best_model_number)), r'Hyperspectral_imaging_zyushi_03\henkango\NN\learning/best_model.h5')
    # loaded_model = tf.keras.models.load_model(str(best_model_number)+'otameshi.h5')
    

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))