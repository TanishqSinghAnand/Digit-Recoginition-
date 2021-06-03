from sys import version
import cv2
import numpy as np
import pandas as pd
import seaborn 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os,ssl,time

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context


X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())

classes = ['0','1','2','3','4','5','6','7','8','9']
nClasses = len(classes)

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scale = X_train/255 
X_test_scale = X_test/255

cls = LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scale,y_train)

y_pred = cls.predict(X_test_scale)
accuracy = accuracy_score(y_test,y_pred)

print(accuracy)

cap = cv2.VideoCapture(0)

while (True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height , width = gray.shape
        upperLeft = (int(width/2-56),int(height/2-56))
        bottomRight = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        im_pil = Image.fromarray(roi)
        img_vb = im_pil.convert('L')
        img_resize = img_vb.resize((28,28),Image.ANTIALIAS)
        img_size_inverted = PIL.ImageOps.invert(img_resize)        
        pixel_filer = 20
        min_pixel = np.percentile(img_size_inverted,pixel_filer)
        img_wb_resize_inverted_scaled = np.clip(img_size_inverted-min_pixel,0,255)
        max_pixel = np.max(img_size_inverted)
        img_wb_resize_inverted_scaled = np.asarray(img_wb_resize_inverted_scaled)/max_pixel
        test_sample = np.asarray(img_wb_resize_inverted_scaled).reshape(1,784)
        test_pred = cls.predict(test_sample)
        print("Test pred = " , test_pred)
        cv2.imshow('frame',gray)
        if(cv2.waitKey(1)&0xff==ord('q')):
            break
    except Exception as e :
        pass

cap.release()
cv2.destroyAllWindows()