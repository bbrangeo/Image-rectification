import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import math
import numpy as np

options ={
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'threshold': 0.5,
    'gpu' : 0.6
    }


tfnet = TFNet(options)

capL = cv2.VideoCapture(0)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
capR = cv2.VideoCapture(1)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
XL=0
distance =0
count = 0
subtracted_image_mean_pre = 5000

while True:
    retL, frameL  = capL.read(cv2.IMREAD_COLOR)
    retR, frameR  = capR.read(cv2.IMREAD_COLOR)

    if (retL == True) and (retR == True):subtracted_image_mean
        resultL = tfnet.return_predict(frameL)
        #resultR = tfnet.return_predict(frameR)

        if not resultL == []:
            for res in resultL:
    
                tl = (res['topleft']['x'], res['topleft']['y'])
                br = (res['bottomright']['x'], res['bottomright']['y'])
                label = res['label']

                frameL = cv2.rectangle(frameL, tl, br, (255, 0 , 0), 5)
                frameL = cv2.putText(frameL, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)

                if label == 'bottle':
                    XL = res['topleft']['x']
                    print("XL:"+str(tl)+'\t' + str(br))
                    #1st y and then
    
                    croped_imageL = frameL[res['topleft']['y']:res['bottomright']['y'],res['topleft']['x']:res['bottomright']['x'],:] 
                    croped_imageL = cv2.cvtColor(croped_imageL,cv2.COLOR_RGB2GRAY)  
                    gray_scale_R  = cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)
                    print(frameL.shape)
                    for i in range(frameL.shape[1]):
                        croped-imageR = gray_scale_R[res['topleft']['y']:res['bottomright']['y'],res['topleft']['x']:res['bottomright']['x'],:]
                        subtracted_image = np.abs(croped_imageR - croped_imageL)
                        subtracted_image_mean = np.sum(subtracted_image,axis = 1)
                        if subtracted_image_mean_pre > subtracted_image_mean:
                            count = i


                    


        """
        if not resultR == []:
            for res in resultR:
        
                tl = (res['topleft']['x'], res['topleft']['y'])
                br = (res['bottomright']['x'], res['bottomright']['y'])
                label = res['label']

                frameR = cv2.rectangle(frameR, tl, br, (255, 0 , 0), 5)
                frameR = cv2.putText(frameR, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)
                if label == 'bottle':
                    XR = res['topleft']['x']
                    print("XR:"+str(XR)+('\n'))
        

        
    disparity = abs(XL-XR)
    x = frameL.shape[0]
    print("x:"+str(x)+"dis:"+str(disparity))

    if not disparity == 0:
        distance = (13*x)/(math.tan(math.radians(30))*disparity)

        

    print("distance"+str(distance))
    """
    cv2.imshow('Left_camera',frameL)
    cv2.imshow('Righr_camera',frameR)
    cv2.imshow('croped',croped_imageL)
       
    K = cv2.waitKey(10)
    if K == 27:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        break       
        

            
