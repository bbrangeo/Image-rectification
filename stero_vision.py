import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import math
import numpy as np
from camera_calibrate import StereoCalibration
cal = StereoCalibration('')
data, stere_par, map = cal.read_data()

options ={
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'threshold': 0.5,
    'gpu' : 0.65
    }


tfnet = TFNet(options)

capL = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
capR = cv2.VideoCapture(1)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

distance =0
count = 0
subtracted_image_mean_pre = 5000
XR = 0
XL=0
while True:
    retL, frameL  = capL.read(cv2.IMREAD_COLOR)
    retR, frameR  = capR.read(cv2.IMREAD_COLOR)

    imgU1 = cv2.remap(frameL,map["map1x"],map["map1y"], cv2.INTER_LANCZOS4)
    imgU2 = cv2.remap(frameR, map["map2x"], map["map2y"], cv2.INTER_LANCZOS4)

    if (retL == True) and (retR == True):
        resultL = tfnet.return_predict(imgU1)
        resultR = tfnet.return_predict(imgU2)

        if not resultL == []:
            for res in resultL:
    
                tl = (res['topleft']['x'], res['topleft']['y'])
                br = (res['bottomright']['x'], res['bottomright']['y'])
                label = res['label']

                imgU1 = cv2.rectangle(imgU1, tl, br, (255, 0 , 0), 5)
                imgU1 = cv2.putText(imgU1, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)

                if label == 'bottle':
                    XL = res['topleft']['x']
                    print("XL:"+str(XL)+'\t' + str(br))
                    #1st y and then
    
                    croped_imageL = imgU1[res['topleft']['y']:res['bottomright']['y'],res['topleft']['x']:res['bottomright']['x'],:] 
                    croped_imageL = cv2.cvtColor(croped_imageL,cv2.COLOR_RGB2GRAY)  
                    gray_scale_R  = cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)
                    cv2.imshow('croped',croped_imageL)

        
        if not resultR == []:
            for res1 in resultR:
                tl = (res1['topleft']['x'], res1['topleft']['y'])
                br = (res1['bottomright']['x'], res1['bottomright']['y'])
                label = res1['label']

                imgU2 = cv2.rectangle(imgU2, tl, br, (255, 0 , 0), 5)
                imgU2 = cv2.putText(imgU2, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)
                if label == 'bottle':
                    XR = res1['topleft']['x']
                    print("XR:"+str(XR)+('\n'))
                    differenceL = abs(res['topleft']['x'] - res['bottomright']['x'])
                    print(differenceL)
                    #differenceR = res1['topleft']['x'] - res1['bottomright']['x']
                    
                    for i in range(50): 
                        croped_imageR = gray_scale_R[res['topleft']['y']:res['bottomright']['y'],res1['topleft']['x']+i:res1['topleft']['x']+differenceL+i]
                        print(croped_imageR)
                        print(croped_imageL)
                        cv2.imshow('croped_imageR',croped_imageR)
                        cv2.waitKey(0)
                        subtracted_image = np.abs(np.subtract(croped_imageR ,croped_imageL))
                        print(subtracted_image.flatten())
                        subtracted_image_mean = (np.sum(subtracted_image.flatten(),axis = 0))/ croped_imageL.shape[0]
                        print(subtracted_image_mean)
                        """
                        if subtracted_image_mean_pre > subtracted_image_mean:
                            count = i
                            subtracted_image_mean_pre = subtracted_image_mean
                        """
        

    #print("i"+str(i))    
    disparity1=0
    disparity = abs(XL-XR)
    x = imgU1.shape[0]
    print("x:"+str(x)+"dis:"+str(disparity))

    if not disparity == 0:
        distance = (13*x)/(math.tan(math.radians(30))*disparity)
        distance = distance - 3
        

    print("distance"+str(distance))
    
    cv2.imshow('Left_camera',imgU1)
    cv2.imshow('Righr_camera',imgU2)
   
       
    K = cv2.waitKey(10)
    if K == 27:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        break       
        

            
