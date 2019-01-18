import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options ={
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'threshold': 0.4,
    'gpu' : 0.71
    }


tfnet = TFNet(options)

#cap = cv2.VideoCapture("http://192.168.43.1:8080/video?x.mjpeg")
cap = cv2.VideoCapture(1)

while True:
	ret, frame  = cap.read(cv2.IMREAD_COLOR)
	if ret == True:
		result = tfnet.return_predict(frame)

		if not result == []:	
			for res in result:
				tl = (res['topleft']['x'], res['topleft']['y'])
				br = (res['bottomright']['x'], res['bottomright']['y'])
				label = res['label']
				confidence = res['confidence']


				frame = cv2.rectangle(frame, tl, br, (255, 0 , 0), 5)
				frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)
				frame = cv2.putText(frame,str(confidence), (res['topleft']['x'], res['topleft']['y']-30), cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)


	cv2.imshow('image',frame)
	K = cv2.waitKey(10)
	if K == 27:
		cap.release()
		cv2.destroyAllWindows()
		break	
