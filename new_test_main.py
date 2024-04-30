import cv2
import numpy as np
import os
from time import sleep
import sys
from PIL import Image
import RPi.GPIO as GPIO 
from mfrc522 import SimpleMFRC522 
from adafruit_servokit import ServoKit

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
kit = ServoKit(channels=16)
servo_turned = False
current_angle = kit.servo[8].angle
# Define names associated with face IDs
names = ['None', 'Josh', 'Dom', 'Ilza', 'Gabriel', 'Williams', 'Yetunde', 'Violin', 'Laila', 'Praise', 'Bode', 'Micheal']

# Map RFID to names
rfid_to_name = {
    245453456310: 'Josh',
    423520521598: 'Dom'
}

def ExistingUser():
    reader = SimpleMFRC522()
    print("Please scan your RFID card")
    try:
        id, text = reader.read()
        if id in rfid_to_name:
            print(f"RFID {id} recognized. Proceeding with facial recognition for {rfid_to_name[id]}...")
            # Initialize and start realtime video capture
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)  # set video width
            cam.set(4, 480)  # set video height

            # Define min window size to be recognized as a face
            minW = 0.1 * cam.get(3)
            minH = 0.1 * cam.get(4)

            while True:
                ret, img = cam.read()
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(int(minW), int(minH)),
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                    if (confidence < 100):
                        face_name = names[face_id]
                        confidence_text = "  {0}%".format(round(100 - confidence))
                    else:
                        face_name = "unknown"
                        confidence_text = "  {0}%".format(round(100 - confidence))

                    cv2.putText(img, str(face_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                    # Check if the recognized face matches the RFID tag
                    if face_name == rfid_to_name[id]:
                          if not servo_turned: # If the servo hasn't been turned yet it will unlock the lock if the servo is closed, and lock it if the lock is unlocked. 
                            if (current_angle < 180): # checks if the servo angle is in the unlocked position.
                                kit.servo[8].angle = 180 # Changes the angle to locked
                                servo_turned = True
                            else:
                                kit.servo[8].angle = 0 # Changes the angle to  unlocked.
                                servo_turned = True  
                        # print(f"Identity confirmed: {face_name}")
                    else:
                        print(f"Identity mismatch: found {face_name}, expected {rfid_to_name[id]}")
                        sys.exit()

                cv2.imshow('camera', img)
                k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
        else:
            print('RFID not found in the system.')
            sys.exit()
    finally:
        GPIO.cleanup()
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

def choice():
    while True:
        user_choice = input("Enter 'e' to start, or 'q' to quit: ")
        if user_choice.lower() == 'e':
            ExistingUser()
        elif user_choice.lower() == 'q':
            print("Exiting program.")
            sys.exit()
        else:
            print("Invalid input.")

choice()
