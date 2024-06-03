#ライブラリインポート
import cv2
import numpy as np

#ベース画像の読み込み
pic=cv2.imread(r'20230901_fiber\label2\2_6_7_9_11_14\top100\910.png', flags=cv2.IMREAD_GRAYSCALE)
# pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)

#疑似カラー化_JET
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_JET)
cv2.imwrite('20230901_fiber/color/pseudo_color_jet.jpg',np.array(pseudo_color))
#疑似カラー化_HOT
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HOT)
cv2.imwrite('20230901_fiber/color/pseudo_color_hot.jpg',np.array(pseudo_color))
#疑似カラー化_HSV
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HSV)
cv2.imwrite('20230901_fiber/color/pseudo_color_hsv.jpg',np.array(pseudo_color))
#疑似カラー化_RAINBOW
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_RAINBOW)
cv2.imwrite('20230901_fiber/color/pseudo_color_rainbow.jpg',np.array(pseudo_color))