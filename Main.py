import cv2
import numpy as np
import os
from time import sleep
import sys
from PIL import Image
import RPi.GPIO as GPIO 
from mfrc522 import SimpleMFRC522 


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#indicate id counter
id = 0

#for gpio pin to connect solenoid


names = ['None', 'Josh', 'Dom', 'Ilza', 'Gabriel', 'Williams', 'Yetunde', 'Violin', 'Laila', 'Praise', 'Bode', 'Micheal']
def NewUser():
    reader = SimpleMFRC522() 
    try:
            idNos = [245453456310,423520521598]
            id, text = reader.read()         
            #print(id)
            if id in idNos:
                print(id)
                print("\n [INFO] start face detector ...")
                cam = cv2.VideoCapture(0)
                cam.set(3, 640) # set video widht
                cam.set(4, 480) # set video height
                # Define min window size to be recognized as a face
                minW = 0.1*cam.get(3)
                minH = 0.1*cam.get(4)
                face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

                # For each person, enter one numeric face id
                face_id = input('\n enter user id end and press <Enter> ==>  ')
                print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
                # Initialize individual sampling face count
                count = 0
                while(True):
                    ret, img = cam.read()
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # flip video image vertically
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)

                    for (x,y,w,h) in faces:

                            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                            count += 1

                            # Save the captured image into the datasets folder
                            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                            cv2.imshow('image', img)

                    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                    if k == 27:
                        break
                    elif count >= 100: # Take 30 face sample and stop video
                        break

                # Do a bit of cleanup
                print("\n [INFO] Exiting Program and cleanup stuff")
                cam.release()
                cv2.destroyAllWindows()

                #Face Training
                print("\n [INFO] Please wait...Initializing...")
                # Path for face image database
                path = 'dataset'

                recognizer = cv2.face.LBPHFaceRecognizer_create()
                detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
                # function to get the images and label data
                def getImagesAndLabels(path):
                    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
                    faceSamples=[]
                    ids = []
                    for imagePath in imagePaths:

                        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                        img_numpy = np.array(PIL_img,'uint8')

                        id = int(os.path.split(imagePath)[-1].split(".")[1])
                        faces = detector.detectMultiScale(img_numpy)

                        for (x,y,w,h) in faces:
                            faceSamples.append(img_numpy[y:y+h,x:x+w])
                            ids.append(id)

                    return faceSamples,ids

                print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
                faces,ids = getImagesAndLabels(path)
                recognizer.train(faces, np.array(ids))

                # Save the model into trainer/trainer.yml
                recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

                # Print the numer of faces trained and end program
                print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
                    
            else:
                print('rfid not found')
            print(text) 
    finally:         
            GPIO.cleanup()


def ExistingUser():
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img =cam.read()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    )  
        
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def choice():
    while True:
        choice = input("Enter 'n' for new user, 'e' for existing user, or 'q' to quit: ")
        if choice.lower() == 'n':
            NewUser()
        elif choice.lower() == 'e':
            ExistingUser()
        elif choice.lower() == 'q':
            print("Exiting program.")
            sys.exit()
        else:
            print("Invalid input.")

choice()