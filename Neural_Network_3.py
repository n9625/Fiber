import cv2
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
import time
import glob
import os
from pathlib import Path

def show(img_arr):
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_arr)
    plt.axis ('off')
    plt.show()
    plt.close()
    return

msk = ['2','6','7','9','11','14','mask_other']
# ROOT = Path('datas/20230721')
# name = 'result_490-1600_3layer_300epo_10000pix_batch32_ES_C2-6-7-9-11-14-other'

f_ROI = ROOT / 'ROI'
number1 = os.listdir(f_ROI)
print(number1)
number1 = ['2_6_7_9_11_14']

for n1 in number1:
    print(n1)
    f_ROI2 = f_ROI / n1
    
    res = cv2.imread('{}/NN/{}/NN_hm.png'.format(ROOT, name))
    show(res)

    M = []
    
    for m in msk:
        mask = cv2.imread('{}/001/{}.png'.format(f_ROI2, m),cv2.IMREAD_GRAYSCALE)
        print(m)
        
        blue, green, red, cyan, yellow, magenta, white = 0, 0, 0, 0, 0, 0, 0

        h,w = mask.shape
        for row in range (h):
            for col in range (w):
                if mask[row][col] > 10:
                    if all(res[row][col] == [255,0,0]):
                        blue += 1
                    elif all(res[row][col] == [0,255,0]):
                        green += 1
                    elif all(res[row][col] == [0,0,255]):
                        red += 1
                    elif all(res[row][col] == [0,255,255]):
                        yellow += 1
                    elif all(res[row][col] == [255,255,0]):
                        cyan += 1
                    elif all(res[row][col] == [255,0,255]):
                        magenta += 1
                    else:
                        white += 1
                else:
                    continue
                    
        print(blue, green, red, cyan, yellow, magenta, white)
        mat = blue, green, red, cyan, yellow, magenta, white
        M.append(mat)
    
    print(M)
    np.savetxt('{}/NN/{}/pixel.csv'.format(ROOT, name), M, fmt='%.10f', delimiter=',')
        