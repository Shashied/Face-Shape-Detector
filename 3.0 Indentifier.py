import pickle
import cv2
import imutils #image utilities library
from imutils import face_utils
import dlib

train_data=pickle.load(open('data.pickle','rb'))
train_target=pickle.load(open('target.pickle','rb'))
print(train_data)
import numpy as np
def predict_face(points,img):

    face_dict={0:'Diamond',1:'Oblong',2:'Oval',3:'Round',4:'Square',5:'Triangle'}
    
    my_points=points[2:9,0] #assgining point 3-point 9 to my points

    d1=my_points[6]-my_points[0]
    d2=my_points[6]-my_points[1]
    d3=my_points[6]-my_points[2]
    d4=my_points[6]-my_points[3]
    d5=my_points[6]-my_points[4]
    d6=my_points[6]-my_points[5]

    D1=d2/float(d1)*100
    D2=d3/float(d1)*100
    D3=d4/float(d1)*100
    D4=d5/float(d1)*100
    D5=d6/float(d1)*100

    result=clsfr.predict([[D1,D2,D3,D4,D5]])
    results=result[0]
    
    cv2.putText(img,face_dict[results],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)
    cv2.imshow('LIVE',img)
    cv2.waitKey(1)
    

from sklearn.neighbors import KNeighborsClassifier #load KNN classifer

clsfr=KNeighborsClassifier()    #KNN classifier is loaded to clsfr

clsfr.fit(train_data,train_target)




camera=cv2.VideoCapture(0)  #loading the default(0) camera into camera object



face_detector=dlib.get_frontal_face_detector()
#loading a face detecting classifer from dlib library into face_detector object

landmarks_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#loading the 68 landmarks detecting classifier into landmarks_detector



while(True):

    ret,img=camera.read()       #reading one frame from camera and assign it to img  
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting the rgb image into a gray image 

    rect=face_detector(gray)    #sending the gray image into face_detector Algorithm
                                #rect contains x1,y1,x2,y2 points of faces
    for rects in rect:
         x1=rects.left()
         y1=rects.top()
         x2=rects.right()
         y2=rects.bottom()

         cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
         cv2.rectangle(img,(x1-1,y1-20),(x1+60,y1),(0,0,255),-1)
         cv2.putText(img,'Face',(x1+3,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    try:
        points=landmarks_detector(gray,rect[0])  #receiving 68 points in face to points object
        points=face_utils.shape_to_np(points) #converting the points into a numpy array

        for p in points:

            cv2.circle(img,(p[0],p[1]),2,(0,255,255),-1)
        
        predict_face(points,img)     

    except Exception as e:

        print(e)
