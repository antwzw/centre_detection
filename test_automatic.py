import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
import pandas as pd
import math

SOURCE_PATHnew = '/Users/zhengbaoqin/Desktop/code/photo/1203/part2-resize/*.jpg'
PROTO_PATH = "/Users/zhengbaoqin/Desktop/code/deploy.prototxt"
MODEL_PATH = "/Users/zhengbaoqin/Desktop/code/hed_pretrained_bsds.caffemodel"
type=38
class CropLayer(object):
    def __init__(self, params, blobs):
        # 初始化剪切区域开始和结束点的坐标
        self.xstart = 0
        self.ystart = 0
        self.xend = 0
        self.yend = 0

    # 计算输入图像的体积
    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # 计算开始和结束剪切坐标的值
        self.xstart = int((inputShape[3] - targetShape[3]) // 2)
        self.ystart = int((inputShape[2] - targetShape[2]) // 2)
        self.xend = self.xstart + W
        self.yend = self.ystart + H

        # 返回体积，接下来进行实际裁剪
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

def HED(image, net, downW=800, downH=800):
    (H, W) = image.shape[:2]
    image = cv2.resize(image, (downW, downH))
    # 根据输入图像为全面的嵌套边缘检测器（Holistically-Nested Edge Detector）构建一个输出blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(800, 800),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=True)
    # # 设置blob作为网络的输入并执行算法以计算边缘图
    #print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    # 调整输出为原始图像尺寸的大小
    hed = cv2.resize(hed[0, 0], (W, H))
    # 将图像像素缩回到范围[0,255]并确保类型为“UINT8”
    hed = (255 * hed).astype("uint8")
    return hed

def crop(image,cx,cy):
    size = image.shape
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    xcenter=corners[38][0][0]
    ycenter=corners[38][0][1]

    # y_start=math.ceil(ycenter-(size[0]-ycenter)/cy)
    # y_end=math.ceil(ycenter+(size[0]-ycenter)/cy)
    # x_start=math.ceil(xcenter-(size[1]-xcenter)/cx)
    # x_end=math.ceil(xcenter+(size[1]-xcenter)/cx)
    cropped = image[math.ceil(ycenter-(size[0]-ycenter)/cy):math.ceil(ycenter+(size[0]-ycenter)/cy),math.ceil(xcenter-(size[1]-xcenter)/cx):math.ceil(xcenter+(size[1]-xcenter)/cx)]
    # cv2.imwrite(RESULT_PATH.format(num = count+1), cropped)
    # print("Crop Finished")
    return cropped

def local_grid(gray, param):
    # detect the center of bolts relative to the centre of
    (H, W) = gray.shape[:2]
    circleshed = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=100, param2=param,
                                 minRadius=30, maxRadius=40)
    if(circleshed is None):
        print('circleshed is None')
    # use K-means algorithm to group the circles and extract the minimum from each groups
    '''
    km = KMeans(n_clusters=4).fit(circleshed[0, :, 0:2])
    # km = circleshed[0,:,0:2]
    df = pd.DataFrame({'X': circleshed[0, :, 0],
                       'Y': circleshed[0, :, 1],
                       'Radius': circleshed[0, :, 2],
                       'Label': km.labels_}
                      )
    mins = df.sort_values('Y', ascending=False).groupby('Label', as_index=False).first()
    '''
    if(circleshed is not None):
        df = pd.DataFrame({'X': circleshed[0, :, 0],
                           'Y': circleshed[0, :, 1],
                           'Radius': circleshed[0, :, 2]})
        df["Distance"]=(df["Y"]-H/2)**2+(df["X"]-W/2)**2
        df=df[df["Distance"]>=10000]
        #参数 我认为这表示的是放几个进去
        a=6
        mins = df.sort_values('Distance', ascending=False).tail(a)
    # print(mins)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x_sum = 0
    y_sum = 0
    # show the circles
    for index, i in mins.iterrows():
        center = (i['X'].astype("int"), i['Y'].astype("int"))
        cv2.circle(result, center, 1, (255, 255, 255), 3)
        radius = i['Radius'].astype("int")
        cv2.circle(result, center, radius, (255, 255, 255), 3)
        # x_sum = x_sum + i['X'].astype("float64")
        # y_sum = y_sum + i['Y'].astype("float64")
        x_sum = x_sum + i['X']
        y_sum = y_sum + i['Y']
    x_sum = x_sum + 1
    # print('measured center: (', x_sum / 4, ",", y_sum / 4,")")

    return result,x_sum,y_sum

def detect_bolts(img, net):
    size = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 按真实长度缩放
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    
    center = corners[type]
    x38=corners[38]
    x38x=x38[0][0]
    x38y=x38[0][1]
    x16=corners[16]
    x16x=x16[0][0]
    x16y=x16[0][1]
    #print('38:',x38x,' ',x38y)
    #print('16:',x16x,' ',x16y)
    
    # print('actual center: ', center)
    center_x = center[0][0]
    center_y = center[0][1]
    zeropoint = corners[0]
    center_xx = zeropoint[0][0]
    center_yy = zeropoint[0][1]
    #print('初始值： （',"%.2f"  % center_xx,",", "%.2f" % center_yy,")")
    #print("actual center: (", "%.2f" % center_x,",", "%.2f" % center_y,")")

    # 求螺栓中心坐标
    gray2 = HED(img,net)
    result,x_sum,y_sum = local_grid(gray2, 18)
    x_diff = x_sum / 4 - center_x
    y_diff = y_sum / 4 - center_y
    a=(x_sum/4-center_xx)/15
    b=(y_sum/4-center_yy)/15
    print('棋盘格位置:', a,' ',b)
    a=(center_x-center_xx)/15
    b=(center_y-center_yy)/15
    print('认为的中心:', a,' ',b)
    diff = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    # difference between measured center and actual center
    # print(diff)
    # print('difference: ')
    # print("%.2f" % diff)
    return result,diff

if __name__ == '__main__':
    cv2.dnn_registerLayer("Crop", CropLayer)
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    images = [cv2.imread(file) for file in glob.glob(SOURCE_PATHnew)]
    difference = []
    for count, image in enumerate(images):
        cx=2
        cy=2
        cropped=crop(image,cx,cy)
        # cv2.imwrite(RESULT_PATH.format(num = count+1), cropped)
        # result = detect_bolts(image,net)
        result,diff = detect_bolts(cropped, net)
        print(diff)
        difference.append(diff)
        # print(diff)
        # cv2.imwrite(RESULT_PATH.format(num = count+1), result)
    difference = np.array(difference)
    print(np.mean(difference))
    print(np.var(difference))
    # print(type(difference))
    print('Finished.')
