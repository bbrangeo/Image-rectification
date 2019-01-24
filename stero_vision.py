import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import math

options ={
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'threshold': 0.5,
    'gpu' : 0.7
    }


tfnet = TFNet(options)

capL = cv2.VideoCapture(0)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
capR = cv2.VideoCapture(1)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
XL=0
XR=0
distance =0

while True:
    retL, frameL  = capL.read(cv2.IMREAD_COLOR)
    retR, frameR  = capR.read(cv2.IMREAD_COLOR)

    if (retL == True) and (retR == True):
        resultL = tfnet.return_predict(frameL)
        resultR = tfnet.return_predict(frameR)

        if not resultL == []:
            for res in resultL:
    
                tl = (res['topleft']['x'], res['topleft']['y'])
                br = (res['bottomright']['x'], res['bottomright']['y'])
                label = res['label']

                frameL = cv2.rectangle(frameL, tl, br, (255, 0 , 0), 5)
                frameL = cv2.putText(frameL, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)
                if label == 'bottle':
                    XL = res['topleft']['x']
                    print("XL:"+str(XL)+'\t')
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
        distance = distance -4

        

    print("distance"+str(distance))
    
    cv2.imshow('Left_camera',frameL)
    cv2.imshow('Righr_camera',frameR)
       
    K = cv2.waitKey(10)
    if K == 27:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        break       
        

            
