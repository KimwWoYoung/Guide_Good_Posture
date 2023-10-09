import cv2
import numpy as np
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

cap = cv2.VideoCapture(0)
 
# Curl counter variables
warning = False
count = 0
start = time.gmtime(time.time())     # 시작 시간 저장


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        resize_frame = cv2.resize(frame ,None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) 
        
        # Recolor image to RGB
        image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            # Calculate angle
            angle = calculate_angle(left_shoulder, right_shoulder)
            print("angle : " + str(angle))
            
            # Curl counter logic
            if angle < 170:
                count = count + 1
            else:
                count = 0
    
        except:
            pass
        
        cv2.rectangle(image, (0,0), (1500,80), (128,128,128), -1)
        
        #Time
        now = time.gmtime(time.time())
        cv2.putText(image, 'Time', 
                    (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        hour = now.tm_hour - start.tm_hour
        minutes = abs(now.tm_min - start.tm_min)
        cv2.putText(image, str(hour) +' : '+ str(minutes), 
                    (20,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        if minutes >= 1:
            cv2.putText(image, 'Stand up and Stretch ', 
                    (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 1, cv2.LINE_AA)

        #Warning                       
        if minutes < 1 and count > 5:
            cv2.putText(image, 'Please Straighten up', 
                    (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 1, cv2.LINE_AA)
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()