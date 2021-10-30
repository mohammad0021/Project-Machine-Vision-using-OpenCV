#!/usr/bin/python3
import os
import cv2 as cv
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image as pitImg
from PIL import ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import time

import joblib
import Sliding as sd
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression


face_cascade_defult= cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
face_cascade_train= cv.CascadeClassifier('models/MyTrainFace.xml')
FaceModelSVM = joblib.load('models/FaceModelSVM.model');

fullbody_cascade_defult= cv.CascadeClassifier('models/haarcascade_fullbody.xml')
fullbody_cascade= cv.CascadeClassifier('models/MyTrainFullBody.xml')
BodyModelSVM = joblib.load('models/BodyModelSVM.model');



recognizer = cv.face.LBPHFaceRecognizer_create()

# Traking
sift = cv.xfeatures2d.SIFT_create()
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)

modes= {
    "FaceDefult": False,
    "FaceTrain": False,
    "BodyDefult": False,
    "BodyTrain": False,
    "FaceSVM": False,
    "BodySVM": False,
    "FaceRecognition": False
}

options={
    "SaveVideo": False,
    "LearningKN": False,
    "ShowHistogram": False,
    "AutoCLAHE": False,
    "TrackingBody": False,

    "GaussianBlur": False,
    "MedianBlur": False,
    "CLAHE": False,
    "Equalize": False,
}


gui = Tk()
gui.wm_title("Video Processing - Mohammad Reza Sajadi")
gui.geometry("1280x670") # 1280x620
gui.configure(bg='#202637')
gui.resizable(width=0, height=0)

def _quit():
    messagebox.showinfo("Thanks", "thanks for your attention")
    gui.quit()     # stops mainloop
    gui.destroy()  # this is necessary on Windows to prevent


# Main video frames
mainFrame = Frame(gui)
mainFrame.place(x=15, y=10) 
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)  
Label(gui, text = " Original video ", background='#343A40', foreground="white").place(x=15, y=15) 

mainFrame2 = Frame(gui)
mainFrame2.place(x=530, y=10)   
lmain2 = tk.Label(mainFrame2)
lmain2.grid(row=50, column=30)        
Label(gui, text = " Processed Video - Detection", background='#343A40', foreground="white").place(x=530, y=15) 
fps_detect = StringVar() 
Label(gui, textvariable=fps_detect, background='#343A40', foreground="white").place(x=530, y=40)  
count_frame = StringVar() 
Label(gui, textvariable=count_frame, background='#343A40', foreground="white").place(x=530, y=65)  


model_select = StringVar() 
Label(gui, textvariable=model_select, background='#343A40', foreground="white").place(x=530, y=90)  


font = cv.FONT_HERSHEY_SIMPLEX

frame_gray=False
prev_frame_time=0
new_frame_time=0
mNeig=100;
frame_objects={}
bmNeig=0;
counter_frame=1;
counter_hist=100

counter_svm=5


size = (64,128)
step_size = (9,9)
downscale = 1.25
scale = 0
detections = []

userid_face=1000;
list_face=[]

# traking
kp_image=False
desc_image=False
body_traking=False

# Form Reg face
FormRFaces= FALSE;


def video_stream(addr=0):
    global prev_frame_time, new_frame_time
    prev_frame_time, new_frame_time= (0,0)

    # addr="videos/1.mp4"
    cap= cv.VideoCapture(addr)
    
    if options["SaveVideo"]:
        fourcc = cv.VideoWriter_fourcc('X','V','I','D')
        output_video= cv.VideoWriter('output.mp4', fourcc, 20.0, (640,480), True)

    def play_video():
        global prev_frame_time, new_frame_time, frame_gray, frame_objects, mNeig, bmNeig, counter_frame
        global counter_hist, userid_face,list_face, kp_image, desc_image, body_traking, FormRFaces, counter_svm

        ret, main_frame = cap.read()

        if not ret: return;
        main_frame= cv.resize(main_frame, (265,265))
        frame= cv.cvtColor(main_frame, cv.COLOR_BGR2RGBA)
        frame_gray= cv.cvtColor(main_frame, cv.COLOR_BGR2GRAY)
        frame_gray2= frame_gray
        
        frame_processed= frame

        new_frame_time = time.time()
        fps_frame = 1/(new_frame_time-prev_frame_time)
        if(prev_frame_time!=new_frame_time):
            fps_detect.set("FPS: "+str(int(fps_frame)))
            prev_frame_time = new_frame_time

        counter_frame+=1
        count_frame.set("Frame: "+str(counter_frame) )


        #########   Preprocessing    #########
        if options["GaussianBlur"]:
            frame_gray= cv.GaussianBlur(frame_gray, (5,5), 0.5)
        if options["MedianBlur"]:
            frame_gray= cv.medianBlur(frame_gray, 3)

        if options["CLAHE"]:
            clahe= cv.createCLAHE(clipLimit=2.0,  tileGridSize=(8,8))
            if options["AutoCLAHE"]:
                mean_light= int(frame_gray.mean())
                if mean_light <= 25:
                    clahe= cv.createCLAHE(clipLimit=9.0,  tileGridSize=(8,8))
                elif mean_light <= 35:
                    clahe= cv.createCLAHE(clipLimit=7.0,  tileGridSize=(8,8))
                elif mean_light <= 40:
                    clahe= cv.createCLAHE(clipLimit=4.0,  tileGridSize=(8,8))

            frame_gray= clahe.apply(frame_gray)

        if options["Equalize"]:
            frame_gray= cv.equalizeHist(frame_gray)
    


        ###############    Models   ###############
        if modes["FaceDefult"]:
            frame_objects["FaceDefult"]= {"objects": face_cascade_defult.detectMultiScale(frame_gray, 1.1, 10) }
            if count_frame_detect.get()==0 and frame_objects["FaceDefult"]["objects"] != ():
                count_frame_detect.set(str(counter_frame))

        if modes["FaceTrain"]:
            frame_objects["FaceTrain"]= {"objects": face_cascade_defult.detectMultiScale(image=frame_gray, scaleFactor=1.07, minNeighbors=mNeig) } # face_cascade_train
            if options["LearningKN"]:
                if count_frame_detect.get()==0 and frame_objects["FaceTrain"]["objects"] != ():
                    count_frame_detect.set(str(counter_frame))

                if frame_objects["FaceTrain"]["objects"] == () and mNeig>5:
                    mNeig-=5
                elif mNeig <=200 and frame_objects["FaceTrain"]["objects"]!= ():
                    mNeig+=5

                if(mNeig != bmNeig):
                    bmNeig=mNeig
                    Kn_body.set("K Value: "+str(mNeig))
                    if( mNeig < MinKn_body.get()): MinKn_body.set(mNeig)
                    elif( mNeig > MaxKn_body.get()): MaxKn_body.set(mNeig)


        if modes["FaceSVM"]:
            counter_svm+=10
            for (x, y, window) in sd.sliding_window(frame_gray, size, step_size):
                if window.shape[0] != size[1] or window.shape[1] != size[0]:
                    continue
                fd= hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
                fd = fd.reshape(1, -1)
                pred = FaceModelSVM.predict(fd)
                if pred == 1:
                    if FaceModelSVM.decision_function(fd) > 0.5:
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), FaceModelSVM.decision_function(fd), 
                        int(size[0] * (downscale**scale)),
                        int(size[1] * (downscale**scale))))
                
                    if count_frame_detect.get()==0:
                        count_frame_detect.set(str(counter_frame))

            clone = frame_gray.copy()
            rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
            sc = [score[0] for (x, y, score, w, h) in detections]
            sc = np.array(sc)
            pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
            frame_objects["FaceSVM"].update( { "objects": pick } )


        if modes["FaceRecognition"] and modes["FaceDefult"] and frame_objects["FaceDefult"]["objects"] != (): #  or frame_objects["FaceTrain"]["objects"] != ()
            for (x,y,w,h) in frame_objects["FaceDefult"]["objects"]:
                face_crop= frame_gray[y:y+h,x:x+w]

                frame_objects["FaceRecognition"]["FaceID"]= "Face validation..."
                faceMatching=False
                if userid_face!=1000:
                    face_id, conf= recognizer.predict(face_crop)
                    if conf <= 70:
                        faceMatching=True
                        frame_objects["FaceRecognition"]["FaceID"]= str(face_id)
                        list_face=[]
                        # print(conf)
                size_capcher=30
                if len(list_face)<size_capcher and not faceMatching:
                    list_face.append(face_crop)
                else:
                    if not faceMatching:  
                        sid= np.full(size_capcher, userid_face)
                        recognizer.train(list_face, sid)
                        baisy=35; baisx=8
                        profile= frame[y-baisy-20:y+h+baisy, x-baisx:x+w+baisx]
                        FormRFaces.add(profile, userid_face)
                        userid_face+=5
                    list_face=[]

        if modes["BodyTrain"]:
            frame_objects["BodyTrain"].update( {"objects": fullbody_cascade.detectMultiScale(image=frame_gray, scaleFactor=1.07, minNeighbors=mNeig)} )
            if options["LearningKN"]:
                if count_frame_detect.get()==0 and frame_objects["BodyTrain"]["objects"] != ():
                    count_frame_detect.set(str(counter_frame))

                if frame_objects["BodyTrain"]["objects"] == () and mNeig>5:
                    mNeig-=5
                elif mNeig <=200 and frame_objects["BodyTrain"]["objects"]!= ():
                    mNeig+=5

                if(mNeig != bmNeig):
                    bmNeig=mNeig
                    Kn_body.set("K Value: "+str(mNeig))
                    if( mNeig < MinKn_body.get()): MinKn_body.set(mNeig)
                    elif( mNeig > MaxKn_body.get()): MaxKn_body.set(mNeig)

        if modes["BodySVM"] and counter_frame==counter_svm:
            counter_svm+=10
            for (x, y, window) in sd.sliding_window(frame_gray, size, step_size):
                if window.shape[0] != size[1] or window.shape[1] != size[0]:
                    continue
                fd= hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
                fd = fd.reshape(1, -1)
                pred = BodyModelSVM.predict(fd)
                if pred == 1:
                    if BodyModelSVM.decision_function(fd) > 0.5:
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), BodyModelSVM.decision_function(fd), 
                        int(size[0] * (downscale**scale)),
                        int(size[1] * (downscale**scale))))
                
                    if count_frame_detect.get()==0:
                        count_frame_detect.set(str(counter_frame))

            clone = frame_gray.copy()
            rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
            sc = [score[0] for (x, y, score, w, h) in detections]
            sc = np.array(sc)
            pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
            frame_objects["BodySVM"].update( { "objects": pick } )


        if options["TrackingBody"] and body_traking is not False:
            kp_grayframe, desc_grayframe = sift.detectAndCompute(frame_gray, None)
            matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            if len(good_points) > 10:
                query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()
                h, w = body_traking.shape
                pts = np.float32([ [0,0],[0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts, matrix)
                frame_gray = cv.polylines(frame_gray, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                # Perspective transform
                h, w = body_traking.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, matrix)
                homography = cv.polylines(frame_processed, [np.int32(dst)], True, (255, 0, 0), 3)





        ###############  Rectangle   ###############
        for key in frame_objects:
            for (x,y,w,h) in frame_objects[key]["objects"]:
                frame_processed= cv.rectangle(frame.copy(), (x,y),(x+w,y+h),(255,255,0), 2)
                if modes["FaceRecognition"] and key=="FaceDefult":
                    key= frame_objects["FaceRecognition"]["FaceID"]
                cv.putText(frame_processed, key, (x,y-5), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        



        ###############   Options   ###############
        if options["ShowHistogram"] and counter_frame==counter_hist:
            counter_hist+=50
            ax_light_his1.clear();
            ax_light_his1.hist(frame_gray2.flatten(), 256, [0,255],  color='r');
            fig_light_his1.canvas.draw()
            fig_light_his1.canvas.flush_events()
            if options["CLAHE"]:
                ax_light_his2.clear()
                ax_light_his2.hist(frame_gray.flatten(), 256, [0,255], color='r');
                fig_light_his2.canvas.draw()
                fig_light_his2.canvas.flush_events()
        
        if options["SaveVideo"]:
            output_video.write(frame_gray)




        ###############  Show Video in windows  ###############
        img= pitImg.fromarray(frame).resize((500, 450))
        imgtk = ImageTk.PhotoImage(image = img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        frame_detect= pitImg.fromarray(frame_processed).resize((500, 450))
        imgtk2= ImageTk.PhotoImage(image = frame_detect)
        lmain2.imgtk = imgtk2
        lmain2.configure(image=imgtk2)
        lmain2.after(10, play_video)

    play_video()

    if options["SaveVideo"]:
        output_video.release()





def browseFiles():
    filename = filedialog.askopenfilename(initialdir= "/", title= "Select a File",filetypes = (("Text files", "*.mp4*"), ("all files", "*.*")))      
    video_stream(filename)


def check_models():
    global frame_objects, FormRFaces;
    modes["FaceDefult"]= FaceDefult.get()
    modes["FaceTrain"]= FaceTrain.get()
    modes["BodyDefult"]= BodyDefult.get()
    modes["BodyTrain"]= BodyTrain.get()
    modes["FaceSVM"]= FaceSVM.get()
    modes["BodySVM"]=         BodySVM.get()
    modes["FaceRecognition"]= FaceRecognition.get()
    

    text_title=""
    if modes["FaceDefult"]: text_title+="Cascade: Face - Defult"    
    if modes["BodyTrain"]:
        text_title+="Cascade: Body - Train"
        frame_objects={"BodyTrain":{}}
        frame_objects["BodyTrain"]["color"]= [255,255,255]
        frame_objects["BodyTrain"]["objects"]= []

    if modes["FaceTrain"]: text_title+=" Face - Train"    
    if modes["BodyDefult"]:  text_title+=" Body - Defult"

    if modes["FaceSVM"]:
        text_title+=" Face - SVM"
        frame_objects={"FaceSVM": {}}
        frame_objects["FaceSVM"]["objects"]= []

    if modes["BodySVM"]:
        text_title+=" Body - SVM"
        frame_objects={"BodySVM": {}}
        frame_objects["BodySVM"]["color"]= [255,255,255]
        frame_objects["BodySVM"]["objects"]= []

    if modes["FaceRecognition"]:
        text_title+=" FaceRecognition"
        frame_objects={"FaceRecognition": {}}
        frame_objects["FaceRecognition"]["objects"]= []
        frame_objects["FaceRecognition"]["FaceID"]= "Face validation..."

        if not FormRFaces:
            FormRFaces= FormRecognizedFaces()

    if not modes["FaceRecognition"]:
        FormRFaces=False



    model_select.set(text_title)


def check_options():
    options["SaveVideo"]= SaveVideo.get()
    options["LearningKN"]= LearningKN.get()
    options["ShowHistogram"]= ShowHistogram.get()
    options["AutoCLAHE"]= AutoCLAHE.get()
    options["TrackingBody"]= TrackingBody.get()

    options["GaussianBlur"]= sGaussianBlur.get()
    options["MedianBlur"]= sMedianBlur.get()
    options["CLAHE"]= sCLAHE.get()
    options["Equalize"]= sEqualize.get()
    
def SelectROI():
    global body_traking,kp_image, desc_image
    x,y,w,h = cv.selectROI(frame_gray, False)
    body_traking= frame_gray[y:y+h,x:x+w]
    kp_image, desc_image = sift.detectAndCompute(body_traking, None)


def Replay():
    pass


frame_menu = Frame(gui)
frame_menu.pack(fill=tk.X, ipady=85, side=tk.BOTTOM)

Button(frame_menu, text="Select Video", command= browseFiles).place(x=20, y=20)  
Button(frame_menu, text="Run WebCame",  command= video_stream).place(x=20, y=55)  
Button(frame_menu, text="Select ROI",  command= SelectROI).place(x=20, y=90)  
Button(frame_menu, text="Replay",  command= Replay).place(x=20, y=125)  
Button(frame_menu, text="Exit",  command= _quit).place(x=110, y=125)  



yop1=35; yop2=65; yop3=95; yop4=125;

FaceDefult= IntVar(); FaceTrain= IntVar(); BodyDefult= IntVar();  BodyTrain= IntVar(); 
BodySVM= IntVar(); FaceSVM= IntVar(); FaceRecognition= IntVar();
Label(frame_menu, text=" --- Models --- ", background='#343A40', foreground="white").place(x=270, y=0)  #280, y=480
xmod1=200; xmod2=330; 
Checkbutton(frame_menu, text=" Face-Defult ", variable=FaceDefult, command=check_models).place(x=xmod1, y=yop1) 
Checkbutton(frame_menu, text=" Face-Train ", variable=FaceTrain,   command=check_models).place(x=xmod2, y=yop1) 
Checkbutton(frame_menu, text=" Body-Defult ", variable=BodyDefult, command=check_models).place(x=xmod1, y=yop2) 
Checkbutton(frame_menu, text=" Body-Train ", variable=BodyTrain,   command=check_models).place(x=xmod2, y=yop2) 
Checkbutton(frame_menu, text=" Face-SVM ", variable=FaceSVM,       command=check_models).place(x=xmod1, y=yop3) 
Checkbutton(frame_menu, text=" Body-SVM ", variable=BodySVM,       command=check_models).place(x=xmod2, y=yop3) 
Checkbutton(frame_menu, text=" Face Recognition ", variable=FaceRecognition,       command=check_models).place(x=xmod1, y=yop4) 


sGaussianBlur= IntVar(); sMedianBlur= IntVar(); sCLAHE= IntVar(); sEqualize= IntVar();
Label(frame_menu, text=" --- Preprocessing --- ", background='#343A40', foreground="white").place(x=530, y=0) 
xpre1=475; xper2=615; 
Checkbutton(frame_menu, text=" Gaussian Blur ", variable=sGaussianBlur, command=check_options).place(x=xpre1, y=yop1)
Checkbutton(frame_menu, text=" Median Blur ", variable=sMedianBlur, command=check_options).place(x=xper2, y=yop1) 
Checkbutton(frame_menu, text=" CLAHE ", variable=sCLAHE, command=check_options).place(x=xpre1, y=yop2) 
Checkbutton(frame_menu, text=" Equalize ", variable=sEqualize, command=check_options).place(x=xper2, y=yop2) 
Checkbutton(frame_menu, text=" 8-Blocking ").place(x=xpre1, y=yop3) 
Checkbutton(frame_menu, text=" 35-Blocking ").place(x=xper2, y=yop3) 

SaveVideo= IntVar(); LearningKN= IntVar(); ShowHistogram= IntVar(); AutoCLAHE= IntVar(); TrackingBody= IntVar();
Label(frame_menu, text=" --- Options --- ", background='#343A40', foreground="white").place(x=850, y=0) 
xop1=770; xop2=910;
Checkbutton(frame_menu, text=" Save Video ", variable=SaveVideo, command=check_options).place(x=xop1, y=yop1) 
Checkbutton(frame_menu, text=" Learning K-N ", variable=LearningKN, command=check_options).place(x=xop2, y=yop1) 
Checkbutton(frame_menu, text=" Show histogram ", variable=ShowHistogram, command=check_options).place(x=xop1, y=yop2) 
Checkbutton(frame_menu, text=" Auto CLAHE ", variable=AutoCLAHE, command=check_options).place(x=xop2, y=yop2) 
Checkbutton(frame_menu, text=" Tracking SIFT ", variable=TrackingBody, command=check_options).place(x=xop1, y=yop3) 





Kn_body= IntVar(); MinKn_body= IntVar(); MaxKn_body= IntVar(); count_frame_detect= IntVar();
MinKn_body.set(100)
count_frame_detect.set(0)
xval=1090; xval2=1200;
Label(frame_menu, text=" --- Values --- ", background='#343A40', foreground="white").place(x=1120, y=0) 
Label(frame_menu, text="K Value: ", background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval, y=yop1) 
Label(frame_menu, textvariable=Kn_body, background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval2, y=yop1) 
Label(frame_menu, text="Max K Value: ", background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval, y=yop2-6) 
Label(frame_menu, textvariable=MaxKn_body, background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval2, y=yop2-6) 
Label(frame_menu, text="Min K Value: ", background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval, y=yop3-12) 
Label(frame_menu, textvariable=MinKn_body, background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval2, y=yop3-12) 

Label(frame_menu, text="Frame Detect: ", background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval, y=yop3+12) 
Label(frame_menu, textvariable=count_frame_detect, background='#343A40', foreground="white", font=("Arial", 10)).place(x=xval2, y=yop3+12) 


# Label(frame_menu, textvariable=Kn_body, background='#343A40', foreground="white").place(x=xval, y=yop1) 
# Label(frame_menu, textvariable=MaxKn_body, background='#343A40', foreground="white").place(x=xval, y=yop2) 
# Label(frame_menu, textvariable=MinKn_body, background='#343A40', foreground="white").place(x=xval, y=yop3) 



##############   Image visualization   ##############
frame_visual = Frame(gui)
frame_visual.pack(fill=tk.X, ipady=200, ipadx=0, side=tk.RIGHT)
Label(frame_visual, text=" Image visualization ", background='#343A40', foreground="white").place(x=40, y=3) 

fig_light_his1 = Figure(figsize=(3, 3), dpi=70)
fig_light_his1.suptitle('Light status')
ax_light_his1= fig_light_his1.add_subplot(111)
ax_light_his1.hist([], 256, [0,255], color='r');
canvas1 = FigureCanvasTkAgg(fig_light_his1, master=frame_visual)  # A tk.DrawingArea.
canvas1.draw()
canvas1.get_tk_widget().grid(row=1, column=0, pady=30, padx=10 , ipadx=0)


fig_light_his2 = Figure(figsize=(3, 3), dpi=70)
fig_light_his2.suptitle('CLAHE hist')
ax_light_his2= fig_light_his2.add_subplot(111)
ax_light_his2.hist([], 256, [0,255], color='r');
canvas2 = FigureCanvasTkAgg(fig_light_his2, master=frame_visual)  # A tk.DrawingArea.
canvas2.draw()
canvas2.get_tk_widget().grid(row=2, column=0, pady=0, padx=10 , ipadx=0)


class FormRecognizedFaces():
    def __init__(self):
        WFRec= Toplevel(gui)
        self.gimage=[]
        self.col=0   
        self.row=0;     
        WFRec.wm_title("Recognized Faces")
        WFRec.geometry("500x500") # 1280x620
        WFRec.configure(bg='#582FBD')
        WFRec.resizable(width=0, height=0)
        Label(WFRec, text="Recognized Faces", font=("Arial", 15)).place( y=10, x=160)
        Button(WFRec, text="Save all to File").place(x=185, y=450)  
        frame_recognized = Frame(WFRec)
        frame_recognized.config(bg="") #
        frame_recognized.grid(row=0, column=0, ipady=80, ipadx=184, pady=45, padx=(20,20))
        self.frame_recognized=frame_recognized
        self.WFRec=WFRec


    def add(self, image, userid):
        image= cv.resize(image,(100,100))
        self.gimage.append( ImageTk.PhotoImage(image=pitImg.fromarray(image)) )
        last= len(self.gimage)

        canvas = Canvas(self.frame_recognized, width = 100, height = 122, bg='black')  
        canvas.create_image( (0, 0), image=self.gimage[last-1], anchor=NW) 
        canvas.grid(row=self.row, column=self.col, padx=(5,5), pady=(5,5), sticky="NW")
        Label(canvas, text="UserID: "+str(userid), font=("Arial", 10)).place(y=102, x=10)
        self.col+=1
        if self.col == 4:
            self.row+=1
            self.col=0


# gui.protocol("WM_DELETE_WINDOW", _quit)
mainloop()