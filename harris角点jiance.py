import numpy as np
import cv2
import math
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from PIL import Image
import pylab as plb
from scipy.ndimage import filters
from PIL import Image
from scipy.ndimage import filters
from numpy import *
from pylab import *
from skimage import io
def get_descriptors(image,filter_coords,wid=5):
    desc=[]
    for coords in filter_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch) # use append to add new elements
    return desc
def match(desc1,desc2,threshold=0.5):
    n = len(desc1[0]) #num of harris descriptors
    #pair-wise distance
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            dot1 = (desc1[i]-mean(desc1[i]))/std(desc1[i])
            dot2 = (desc2[j]-mean(desc2[j]))/std(desc2[j])
            ncc_value = sum(dot1*dot2)/(n-1)
            if ncc_value>threshold:
                d[i,j] = ncc_value

    ndx = argsort(-d)
    matchscores = ndx[:,0]

    return matchscores
def match_twosided(desc1,desc2,threshold=0.5):
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)

    ndx_12 = where(matches_12>=0)[0]
    print (ndx_12.dtype)
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] !=n:
            matches_12[n] = -1
    return matches_12    
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    
    imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')
def appendimages(im1,im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1<rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1<rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    return concatenate((im1,im2),axis=1)
#打开两张图片，并将矩阵各个元素表示成float格式
img1 = io.imread('pic/test.jpg')
img2 = io.imread('pic/test4.jpg')

#图像复制
color_img1=img1.copy()
color_img2=img2.copy()
hog_image=img1.copy()

img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(float)
img2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(float)
#获得图片高度
height1, width1 =img1.shape
height2, width2 =img2.shape
#hog描述子
hog=cv2.HOGDescriptor()
#计算n维数组的梯度，返回和原始数组同样大小的结果。
#两个边界的元素直接用后一个减去前一个值，得到梯度，即b-a；
#对于中间的元素，取相邻两个元素差的一半，即(c-a) / 2。
dy1,dx1=np.gradient(img1)
dy2,dx2=np.gradient(img2)
plt.figure(figsize=(10, 8))
plt.subplot(2,2,1)
plt.imshow(dx1)
plt.subplot(2,2,2)
plt.imshow(dx1, cmap='gray') # cmap = colormap
plt.subplot(2,2,3)
plt.imshow(dy1)
plt.subplot(2,2,4)
plt.imshow(dy1, cmap='gray') # cmap = colormap
plt.show()
#计算Ix^2 Iy^2 IxIy(中间的矩阵)
#plt.imshow(dx);plt.show()
#plt.imshow(dx);plt.show()
Ixx2=dx2**2
Iyy2=dy2**2
Ixy2=dx2*dy2

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(Ixx2, 'gray')
plt.subplot(132)
plt.imshow(Iyy2, 'gray') # cmap = colormap
plt.subplot(133)
plt.imshow(Ixy2, 'gray');plt.show()
#高斯滤波
Ixx2=cv2.GaussianBlur(Ixx2,(3,3),2)
Iyy2=cv2.GaussianBlur(Iyy2,(3,3),2)
Ixy2=cv2.GaussianBlur(Ixy2,(3,3),2)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(Ixx2, 'gray')
plt.subplot(132)
plt.imshow(Iyy2, 'gray') # cmap = colormap
plt.subplot(133)
plt.imshow(Ixy2, 'gray');plt.show()
Ixx1=dx1**2
Ixx1=cv2.GaussianBlur(Ixx1,(3,3),2)
Iyy1=dy1**2
Iyy1=cv2.GaussianBlur(Iyy1,(3,3),2)
Ixy1=dx1*dy1
Ixy1=cv2.GaussianBlur(Ixy1,(3,3),2)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(Ixx1, 'gray')
plt.subplot(132)
plt.imshow(Iyy1, 'gray') # cmap = colormap
plt.subplot(133)
plt.imshow(Ixy1, 'gray');plt.show()
thres=0.01
windowsize=3
rmax=0
#矩阵m的行列式
det= (Ixx1*Iyy1) - (Ixy1**2)
#trace为矩阵m的迹
trace= Ixx1+Iyy1
#角点相应函数R
R=det/trace
R=np.nan_to_num(R)
#对r使用非极大值抑制
map(max, R)
list(map(max, R))
maxim=max(map(max, R))
print(maxim)
#使用最大过滤器过滤输入图像。
localmax=maximum_filter(R,size=3)
plt.imshow(localmax, 'gray');plt.show()

# localFilter= (R == localmax) # 把R和localmax不相等的地方置为零？？？？？？
# 知不知道有一种叫做浮点误差的
localFilter= np.abs(R - localmax) < 1e-10 # 把R和localmax不相等的地方置为零？？？？？？
plt.imshow(localFilter, 'gray');plt.show()

R = R * localFilter
plt.imshow(R, 'gray');plt.show()
#对图2进行相同的操作：

det1= (Ixx2*Iyy2)-(Ixy2**2)
trace1= Ixx2+Iyy2

R1=det1/trace1
R1=np.nan_to_num(R1)
map(max, R1)
list(map(max, R1))
maxim1=max(map(max, R1))
print(maxim1)
localmax1=maximum_filter(R1,size=3)
localFilter1=R1 ==localmax1
R1=R1 * localFilter1
plt.imshow(R1, 'gray');plt.show()
tempTuple1=[]
for y in range(0,height1,2):
    for x in range(0,width1,2):
        if(R[y,x]>(maxim*thres)):
            if x-8<0 or x+8> width1 or y-8<0 or y+8>height1:
                continue;
            ##finalimage[y,x]=r[y,x]
            tempTuple1.append((y,x))
            #设置图像的像素值
            color_img1.itemset((y,x,0),0)
            color_img1.itemset((y,x,1),0)
            color_img1.itemset((y,x,2),255)
            #画出矩阵
            cv2.rectangle(color_img1,(x-2,y-2),(x+2,y+2),255,1)
plt.imshow(color_img1);plt.show()
#vis = np.concatenate((imgo, color_img), axis=1)
#cv2.imshow("Test",vis)
#cv2.waitKey(0)
#cv2.destroyAllWindows();
#对图2进行同样的操作
tempTuple2=[]
for y in range(0,height2,2):
    for x in range(0,width2,2):
        if(R1[y,x]>(maxim1*thres)):
            #finalimage[y,x]=r[y,x]
            if x-8<0 or x+8> width2 or y-8<0 or y+8>height2:
                continue;
            tempTuple2.append((y,x))
            color_img2.itemset((y,x,0),0)
            color_img2.itemset((y,x,1),0)
            color_img2.itemset((y,x,2),255)
            cv2.rectangle(color_img2,(x-2,y-2),(x+2,y+2),255,1)
plt.imshow(color_img2);plt.show()
#设置图片的格式，注意tempTuple是包含图像特征值的点
arr1 = img1.astype(np.float64)
arr2 = img2.astype(np.float64)

d1 = get_descriptors(img1,tempTuple1,5)    
#for q in tempTuple:
#    desDict[(q[0],q[1])]=getdescriptor(arr,q[0],q[1])
d2 = get_descriptors(img2,tempTuple2,5)    
#for e in tempTuple1:
#    desDict1[(e[0],e[1])]=getdescriptor(arr2,e[0],e[1])
print ('starting matching')
matches = match_twosided(d1, d2)
plt.figure(figsize=(10, 8))
plt.gray()
plot_matches(color_img1, color_img2, tempTuple1, tempTuple2, matches)
plt.show()