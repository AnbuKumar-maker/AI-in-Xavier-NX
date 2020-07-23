import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np
width=1280
height=720
#cam=jetson.utils.gstCamera(width,height,'/dev/video1')
cam=jetson.utils.gstCamera(width,height,'0')
net=jetson.inference.imageNet('googlenet')
timeMark=time.time()
fpsFilter=0
timeMark=time.time()
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    frame, width, height = cam.CaptureRGBA(zeroCopy=1)
    classID, confidence = net.Classify(frame, width, height)
    item = net.GetClassDesc(classID)
    dt=time.time()-timeMark
    fps=1/dt
    fpsFilter=.95*fpsFilter+.05*fps
    timeMark=time.time()
    frame=jetson.utils.cudaToNumpy(frame,width,height,4)
    frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR).astype(np.uint8)
    cv2.putText(frame,str(round(fpsFilter,1))+'      '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('webCam',frame)
    cv2.moveWindow('webCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
