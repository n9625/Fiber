import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools


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

def MASKTEKIYOU(mask1):
    data = cv2.imread(r'20230901_fiber\mask\2102.png', flags=cv2.IMREAD_GRAYSCALE)  
    data1 = data.copy()
    data1[:,:] = 0

    n_labels, labels = cv2.connectedComponents(data)

    a1,a2 = np.where(data!=255)
    mask1[a1,a2] = 0

    a1,a2 = np.where(mask1==255)
    l1 = labels[a1,a2]
    # print(sorted(l1, reverse=True))
    l1 = list(set(l1))
    # print(l1)
    # print(labels)
    for t in l1:
        print(t)
        a3,a4 = np.where(labels==t)
        data1[a3,a4]=255
    
    return data1

def SEIKAIRITU(siki, data):
    #  accuracy1 = accuracy_score(data1, siki1)
    # precision1 = precision_score(data1, siki1)
    # recall1 = recall_score(data1, siki1)
    # f_score1 = f1_score(data1, siki1)
    # print(accuracy1, precision1, recall1, f_score1)

    # for f,g in enumerate(index):
    #     # print(f)
    #     del siki1[g-f]
    #     del data1[g-f]
    # print(len(mask_sin))
        
    TN, FP, FN, TP = confusion_matrix(siki, data).ravel()
    print(TN, FP, FN, TP, TN+FP+FN+TP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    specificity = TN/(TN+FP)
    fscore = (2*recall*precision)/(recall+precision)
    # print(accuracy, recall, precision, specificity, fscore)
    return accuracy, recall, precision, specificity, fscore

def main():   
            # for wl in del_wave_length:
            #     print('=====')
            #     print(wl)
            img = cv2.imread(r'20230901_fiber\label2\2_6_7_9_11_14\top100\{}.png'.format(910), flags=cv2.IMREAD_GRAYSCALE)  
            siki = cv2.imread(r'20230901_fiber\NN\test\top100.2\496~1600\NN_hm2_6_7_9_11_14.png', flags=cv2.IMREAD_UNCHANGED) 
            mask_sin = cv2.imread(r'20230901_fiber\mask\2102.png', flags=cv2.IMREAD_GRAYSCALE)  
            zero = img.copy()
            zero[:,:] = 0
            zero1 = zero.copy()
            zero2 = zero.copy()
            zero3 = zero.copy()
            zero4 = zero.copy()
            zero5 = zero.copy()
            zero6 = zero.copy()

            plt.imshow(img)
            plt.show()

            #zyushi
            #2
            # mask_poly1 = np.array([[467,278],[437,308],[449,361],[442,378],[456,391],[436,423],[455,442],[451,460],[483,487],[579,443],[596,451],[613,447],[620,387],[605,372],[614,292],[593,274],[525,274],[505,271],[488,286]])
            mask_poly1 = np.array([[467,278],[437,308],[449,361],[442,378],[456,391],[436,423],[455,442],[451,460],[467,476],[579,443],[596,451],[613,447],[620,387],[605,372],[614,292],[593,274],[525,274],[505,271],[488,286]])
            # mask_poly2 = np.array([[485,509],[466,518],[463,576],[478,589],[502,582],[518,588],[555,575],[570,587],[620,564],[614,500],[578,484],[555,488],[540,482]])  
            mask_poly2 = np.array([[467,495],[485,509],[466,518],[463,576],[478,589],[502,582],[518,588],[555,575],[570,587],[620,564],[614,500],[578,484],[555,488],[540,482]])           
            # mask_poly3 = np.array([[495,599],[475,611],[471,647],[487,661],[479,701],[494,714],[489,734],[504,760],[533,766],[566,776],[627,778],[638,764],[661,766],[645,727],[648,708],[632,672],[642,658],[625,621],[635,603],[622,586],[564,611],[550,597],[509,612]])
            mask_poly3 = np.array([[475,611],[471,647],[487,661],[479,701],[494,714],[489,734],[504,760],[533,766],[566,776],[627,778],[638,764],[661,766],[645,727],[648,708],[632,672],[642,658],[625,621],[635,603]])
            mask_poly4 = np.array([[703,195],[656,219],[644,244],[654,256],[644,335],[659,347],[656,389],[668,402],[665,422],[682,433],[703,428],[720,441],[851,401],[860,344],[864,284],[850,268],[857,229],[808,192],[753,192],[736,202],[716,210]])
            mask_poly5 = np.array([[709,461],[674,476],[693,513],[682,580],[678,605],[757,595],[811,609],[845,582],[831,480],[801,455],[729,472]])
            mask_poly6 = np.array([[712,631],[685,660],[705,735],[711,761],[795,775],[810,759],[848,763],[851,731],[847,630],[828,632],[766,618],[769,638]])
 

            img1 = img.copy()
            img2 = img.copy()
            img3 = img.copy()
            img4 = img.copy()
            img5 = img.copy()
            img6 = img.copy()
            clud1 = img.copy()
            clud2 = img.copy()
            clud3 = img.copy()
            clud4 = img.copy()
            clud5 = img.copy()
            clud6 = img.copy()
            clud7 = img.copy()
            clud1 = cv2.cvtColor(clud1, cv2.COLOR_GRAY2BGR)
            clud2 = cv2.cvtColor(clud2, cv2.COLOR_GRAY2BGR)
            clud3 = cv2.cvtColor(clud3, cv2.COLOR_GRAY2BGR)
            clud4 = cv2.cvtColor(clud4, cv2.COLOR_GRAY2BGR)
            clud5 = cv2.cvtColor(clud5, cv2.COLOR_GRAY2BGR)
            clud6 = cv2.cvtColor(clud6, cv2.COLOR_GRAY2BGR)
            clud7 = cv2.cvtColor(clud7, cv2.COLOR_GRAY2BGR)

            img_mask1 = cv2.fillPoly(img1, [mask_poly1], 255)
            img_mask2 = cv2.fillPoly(img2, [mask_poly2], 255)
            img_mask3 = cv2.fillPoly(img3, [mask_poly3], 255)
            img_mask4 = cv2.fillPoly(img4, [mask_poly4], 255)
            img_mask5 = cv2.fillPoly(img5, [mask_poly5], 255)
            img_mask6 = cv2.fillPoly(img6, [mask_poly6], 255)
            mask1 = cv2.fillPoly(zero1, [mask_poly1], 255)
            mask2 = cv2.fillPoly(zero2, [mask_poly2], 255)
            mask3 = cv2.fillPoly(zero3, [mask_poly3], 255)
            mask4 = cv2.fillPoly(zero4, [mask_poly4], 255)
            mask5 = cv2.fillPoly(zero5, [mask_poly5], 255)
            mask6 = cv2.fillPoly(zero6, [mask_poly6], 255)

            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(1), img_mask1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(2), mask1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(21), img_mask2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(22), mask2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(31), img_mask3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(32), mask3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(41), img_mask4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(42), mask4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(51), img_mask5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(52), mask5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(61), img_mask6)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(62), mask6)

            ####################

            data1 = MASKTEKIYOU(mask1)
            data2 = MASKTEKIYOU(mask2)
            data3 = MASKTEKIYOU(mask3)
            data4 = MASKTEKIYOU(mask4)
            data5 = MASKTEKIYOU(mask5)
            data6 = MASKTEKIYOU(mask6)
            bata1 = cv2.cvtColor(data1, cv2.COLOR_GRAY2BGR)
            bata2 = cv2.cvtColor(data2, cv2.COLOR_GRAY2BGR)
            bata3 = cv2.cvtColor(data3, cv2.COLOR_GRAY2BGR)
            bata4 = cv2.cvtColor(data4, cv2.COLOR_GRAY2BGR)
            bata5 = cv2.cvtColor(data5, cv2.COLOR_GRAY2BGR)
            bata6 = cv2.cvtColor(data6, cv2.COLOR_GRAY2BGR)
            o1,p1,q1 = np.where(bata1==255)
            o2,p2,q2= np.where(bata2==255)
            o3,p3,q3 = np.where(bata3==255)
            o4,p4,q4 = np.where(bata4==255)
            o5,p5,q5 = np.where(bata5==255)
            o6,p6,q6 = np.where(bata6==255)
            bata1[o1,p1] = [255,0,0]
            bata2[o2,p2] = [0,255,0]
            bata3[o3,p3] = [0,0,255]
            bata4[o4,p4] = [255,255,0]
            bata5[o5,p5] = [0,255,255]
            bata6[o6,p6] = [255,0,255]
            clud1[o1,p1] = [255,0,0]
            clud2[o2,p2] = [0,255,0]
            clud3[o3,p3] = [0,0,255]
            clud4[o4,p4] = [255,255,0]
            clud5[o5,p5] = [0,255,255]
            clud6[o6,p6] = [255,0,255]
          
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(101), data1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(102), bata1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(103), clud1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(121), data2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(122), bata2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(123), clud2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(131), data3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(132), bata3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(133), clud3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(141), data4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(142), bata4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(143), clud4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(151), data5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(152), bata5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(153), clud5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(161), data6)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(162), bata6)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(163), clud6)

            data7 = mask_sin.copy()
            bata7 = cv2.cvtColor(data7, cv2.COLOR_GRAY2BGR)
            e1,f1 = np.where(data1!=0)
            e2,f2 = np.where(data2!=0)
            e3,f3 = np.where(data3!=0)
            e4,f4 = np.where(data4!=0)
            e5,f5 = np.where(data5!=0)
            e6,f6 = np.where(data6!=0)
            data7[e1,f1] = 0
            data7[e2,f2] = 0
            data7[e3,f3] = 0
            data7[e4,f4] = 0
            data7[e5,f5] = 0
            data7[e6,f6] = 0
            bata7[o1,p1] = [255,0,0]
            bata7[o2,p2] = [0,255,0]
            bata7[o3,p3] = [0,0,255]
            bata7[o4,p4] = [255,255,0]
            bata7[o5,p5] = [0,255,255]
            bata7[o6,p6] = [255,0,255]
            clud7[o1,p1] = [255,0,0]
            clud7[o2,p2] = [0,255,0]
            clud7[o3,p3] = [0,0,255]
            clud7[o4,p4] = [255,255,0]
            clud7[o5,p5] = [0,255,255]
            clud7[o6,p6] = [255,0,255]
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(171), data7)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(172), bata7)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(173), clud7)

            ####################

            siki1 = siki.copy()
            siki2 = siki.copy()
            siki3 = siki.copy()
            siki4 = siki.copy()
            siki5 = siki.copy()
            siki6 = siki.copy()
            b1,c1,d1 = np.where(siki!=[255,0,0])
            b2,c2,d2 = np.where(siki!=[0,255,0])
            b3,c3,d3 = np.where(siki!=[0,0,255])
            b4,c4,d4 = np.where(siki!=[255,255,0])
            b5,c5,d5 = np.where(siki!=[0,255,255])
            b6,c6,d6 = np.where(siki!=[255,0,255])
            siki1[b1,c1] = [0,0,0]
            siki2[b2,c2] = [0,0,0]
            siki3[b3,c3] = [0,0,0]
            siki4[b4,c4] = [0,0,0]
            siki5[b5,c5] = [0,0,0]
            siki6[b6,c6] = [0,0,0]

            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(200), siki)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(201), siki1)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(202), siki2)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(203), siki3)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(204), siki4)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(205), siki5)
            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(206), siki6)

            siki7 = mask_sin.copy()
            siki1 = cv2.cvtColor(siki1, cv2.COLOR_BGR2GRAY)
            siki2 = cv2.cvtColor(siki2, cv2.COLOR_BGR2GRAY)
            siki3 = cv2.cvtColor(siki3, cv2.COLOR_BGR2GRAY)
            siki4 = cv2.cvtColor(siki4, cv2.COLOR_BGR2GRAY)
            siki5 = cv2.cvtColor(siki5, cv2.COLOR_BGR2GRAY)
            siki6 = cv2.cvtColor(siki6, cv2.COLOR_BGR2GRAY)
            b7,c7 = np.where(siki1!=0)
            b8,c8 = np.where(siki2!=0)
            b9,c9 = np.where(siki3!=0)
            b10,c10 = np.where(siki4!=0)
            b11,c11 = np.where(siki5!=0)
            b12,c12 = np.where(siki6!=0)
            siki7[b7,c7] = 0
            siki7[b8,c8] = 0
            siki7[b9,c9] = 0
            siki7[b10,c10] = 0
            siki7[b11,c11] = 0
            siki7[b12,c12] = 0

            cv2.imwrite(r'20230901_fiber\mask\2_6_7_9_11_14\{}.png'.format(207), siki7)
           
            ####################            
            # print(siki)
            # print(bata1)

            # for f,g in enumerate(index):
            #     print(f)
            #     del mask_sin[g-f]
            # print(len(mask_sin))
            # siki1 = siki1.tolist()
            # list( itertools.chain.from_iterable(siki1))
            # data1 = data1.tolist()
            # list(itertools.chain.from_iterable(data1))
            # siki1 = cv2.cvtColor(siki1, cv2.COLOR_BGR2GRAY)

            n1, n2 = np.where(mask_sin==0)
            t1, t2 = np.where(mask_sin!=0)
            print(n1, n2)
            print(len(n1), len(t1), len(n1)+len(t1))
            mask_sin = mask_sin.flatten().tolist()
            index = [i for i, x in enumerate(mask_sin) if x == 0]
            print(len(index), np.max(index), np.min(index))
         
            x1,z1 = np.where(siki1!=0)
            x2,z2 = np.where(siki2!=0)
            x3,z3 = np.where(siki3!=0)
            x4,z4 = np.where(siki4!=0)
            x5,z5 = np.where(siki5!=0)
            x6,z6 = np.where(siki6!=0)
            x7,z7 = np.where(siki7!=0)
            siki1[x1,z1] = 1
            siki2[x2,z2] = 1
            siki3[x3,z3] = 1
            siki4[x4,z4] = 1
            siki5[x5,z5] = 1
            siki6[x6,z6] = 1
            siki7[x7,z7] = 1
            v1,w1 = np.where(data1!=0)
            v2,w2 = np.where(data2!=0)
            v3,w3 = np.where(data3!=0)
            v4,w4 = np.where(data4!=0)
            v5,w5 = np.where(data5!=0)
            v6,w6 = np.where(data6!=0)
            v7,w7 = np.where(data7!=0)
            data1[v1,w1] = 1     
            data2[v2,w2] = 1  
            data3[v3,w3] = 1  
            data4[v4,w4] = 1  
            data5[v5,w5] = 1  
            data6[v6,w6] = 1   
            data7[v7,w7] = 1    
            siki1 = siki1.flatten().tolist()
            siki2 = siki2.flatten().tolist()
            siki3 = siki3.flatten().tolist()
            siki4 = siki4.flatten().tolist()
            siki5 = siki5.flatten().tolist()
            siki6 = siki6.flatten().tolist()
            siki7 = siki7.flatten().tolist()
            print(type(siki1), len(siki1), 1024*1280)        
            data1 = data1.flatten().tolist()
            data2 = data2.flatten().tolist()
            data3 = data3.flatten().tolist()
            data4 = data4.flatten().tolist()
            data5 = data5.flatten().tolist()
            data6 = data6.flatten().tolist()
            data7 = data7.flatten().tolist()
            print(type(data1), len(data1))
            
            for f,g in enumerate(index):
                print(f)
                del siki1[g-f]
                del siki2[g-f]
                del siki3[g-f]
                del siki4[g-f]
                del siki5[g-f]
                del siki6[g-f]
                del siki7[g-f]
                del data1[g-f]
                del data2[g-f]
                del data3[g-f]
                del data4[g-f]
                del data5[g-f]
                del data6[g-f]
                del data7[g-f]
            # print(len(mask_sin))
            
            accurary1, recall1, precision1, specificity1, fscore1 = SEIKAIRITU(siki1, data1)
            accurary2, recall2, precision2, specificity2, fscore2 = SEIKAIRITU(siki2, data2)
            accurary3, recall3, precision3, specificity3, fscore3 = SEIKAIRITU(siki3, data3)
            accurary4, recall4, precision4, specificity4, fscore4 = SEIKAIRITU(siki4, data4)
            accurary5, recall5, precision5, specificity5, fscore5 = SEIKAIRITU(siki5, data5)
            accurary6, recall6, precision6, specificity6, fscore6 = SEIKAIRITU(siki6, data6)
            accurary7, recall7, precision7, specificity7, fscore7 = SEIKAIRITU(siki7, data7)

            print(accurary1, recall1, precision1, specificity1, fscore1)
            print(accurary2, recall2, precision2, specificity2, fscore2)
            print(accurary3, recall3, precision3, specificity3, fscore3)
            print(accurary4, recall4, precision4, specificity4, fscore4)
            print(accurary5, recall5, precision5, specificity5, fscore5)
            print(accurary6, recall6, precision6, specificity6, fscore6)
            print(accurary7, recall7, precision7, specificity7, fscore7)

            ###################
 

"""3. execution"""
if __name__ == "__main__":
    before_time = time.ctime()
    before_cnvtime = time.strptime(before_time)
    
    main()
    
    after_time = time.ctime()
    after_cnvtime = time.strptime(after_time)
    
    print("start time : ", time.strftime("%Y/%m/%d %H:%M:%S", before_cnvtime))
    print("finish time : ", time.strftime("%Y/%m/%d %H:%M:%S", after_cnvtime))