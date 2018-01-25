# coding: utf-8

# In[3]:


import numpy as np
import cv2


cap = cv2.VideoCapture(0)   #Open camera
out = cv2.VideoWriter('output.MP4',-1, 5.0, (1280,720))   #Write output vedio, not used
firstFrame = None
min_area = 400   #Default contour
i = 0   #Captured image idx
xout = list()   #Output coordinates
yout = list()


while(cap.isOpened()):
    ret, frame = cap.read()   #Capture each frame
    ret = cap.set(3,1280)   # 1280 * 720
    ret = cap.set(4,720)
    ret = cap.set(5,25)   # 25fps
    #np.flip(frame,2)
    
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Convert Color To Grey
        if firstFrame is None:   
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(frame[:,:,1], gray)   #Difference between current frame and original
        #sp.medfilt2d(frameDelta, [3, 3])
        
        thresh = cv2.threshold(frameDelta, 23, 255, cv2.THRESH_BINARY)[1]   #Difference < 23: pixels black; Otherwise white
        thresh = cv2.dilate(thresh, None, iterations=2)   #Highlight white pixels
        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box = np.zeros((4,2))
        k = 0   #Rotation angle of box
        xc = yc =0   #Coordinates of box
        #If 'marker' appears
        for c in cnts:   
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
                
            #Compute the bounding box for the contour    
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            k = - (box[2,1] - box[1,1])/(box[2,0] - box[1,0])
            k = np.arctan(k) / np.pi * 180
            yc = (box[2,0] + box[0,0]) / 2
            xc = (box[2,1] + box[0,1]) / 2
            xout.append(xc)
            yout.append(yc)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)
        
        cv2.putText(img = frame, text = 'theta=' + str(round(k,4)), org = (int(yc)+20,int(xc)+40), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255, 255, 40))
        cv2.putText(img = frame, text = 'yc=' + str(round(yc,4)), org = (int(yc)+20,int(xc)+0), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255, 255, 40))
        cv2.putText(img = frame, text = 'xc=' + str(round(xc,4)), org = (int(yc)+20,int(xc)+20), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255, 255, 40))
        
 
        #Video show
        cv2.imshow("Security Feed", frame)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)

        #Screenshot each single frame from the video
        if cv2.waitKey(1) & 0xFF == ord('s'):
            i = i + 1
            screenshot = 'screenshot' + `i`
            cv2.imshow(screenshot, frame)
        #Exit    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        

xout = np.array(xout)
yout = np.array(yout)
csv = np.asarray([ xout, yout ])
csv = csv.T
np.savetxt("xxx.csv", csv, delimiter=",")

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

