import cv2
import mediapipe as mp
import subprocess

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Windows Code to turn off and on wifi
def turn_on_wifi():
    subprocess.call('netsh interface set interface "Wi-Fi" admin=enabled', shell=True)
    print('Wi-Fi turned on')

def turn_off_wifi():
    subprocess.call('netsh interface set interface "Wi-Fi" admin=disabled', shell=True)
    print('Wi-Fi turned off')

#Linux wifi code 
'''def turn_on_wifi():
    subprocess.call('nmcli radio wifi on', shell=True)
    print('Wi-Fi turned on')

def turn_off_wifi():
    subprocess.call('nmcli radio wifi off', shell=True)
    print('Wi-Fi turned off')'''

motion_detected = False  # Initialize motion detection flag to False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # Apply a threshold to the foreground mask to obtain a binary image
    thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)[1]

    # Detect human motion by checking if a person is present in the frame
    results = mp_holistic.process(frame)
    if results.pose_landmarks:
        motion_detected = True
        turn_on_wifi()
        cv2.putText(frame, "Motion Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Device Status: Turned On", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        motion_detected = False
        turn_off_wifi()
        cv2.putText(frame, "No Motions Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        cv2.putText(frame, "Devives Turned off", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
del mp_holistic
