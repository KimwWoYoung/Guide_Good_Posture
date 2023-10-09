import cv2
import numpy as np

def calculate_angle_using_atan2(A, B, C):
    AB = B - A
    BC = C - B
    theta_AB = np.arctan2(AB[1], AB[0])
    theta_BC = np.arctan2(BC[1], BC[0])
    angle = np.degrees(theta_AB - theta_BC)
    angle = (angle + 360) % 360
    if angle > 180:
        angle = 360 - angle
    return angle

def direction_determination(output):
    left_ear_confidence = output[0, 16, 0, 2]
    right_ear_confidence = output[0, 17, 0, 2]
    if left_ear_confidence > right_ear_confidence:
        return 'left'
    elif right_ear_confidence > left_ear_confidence:
        return 'right'
    else:
        return 'front'

def draw_all_keypoints_on_image(output, frame, w, h):
    num_keypoints = output.shape[1]
    colors = [(0, 255, 0) for _ in range(num_keypoints)]  # 초록색으로 모든 키포인트를 표시
    
    for idx in range(num_keypoints):
        confidence = output[0, idx, 0, 2]
        x = int(output[0, idx, 0, 0] * w)
        y = int(output[0, idx, 0, 1] * h)
        if confidence > 0.01:  # Threshold to filter keypoints
            cv2.circle(frame, (x, y), 5, colors[idx], -1)
    return frame


protoFile_coco = r"C:\Users\rladn\Downloads\openpose-master\openpose-master\models\pose\coco\pose_deploy_linevec.prototxt"
weightsFile_coco = r"C:\Users\rladn\Desktop\opencv\pose_iter_440000.caffemodel"
net_coco = cv2.dnn.readNetFromCaffe(protoFile_coco, weightsFile_coco)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Check camera connection.")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net_coco.setInput(blob)
    output_coco = net_coco.forward()

    # 모델 출력값 확인
    print("Model Output shape:", output_coco.shape)
    print("Sample Keypoint Data for Neck:", output_coco[0, 1, 0])

    direction = direction_determination(output_coco)
    
    # 방향 결정 함수의 결과 확인
    print("Determined Direction:", direction)

    if direction == 'left' and output_coco[0, 16, 0, 2] > 0.01 and output_coco[0, 1, 0, 2] > 0.01 and output_coco[0, 5, 0, 2] > 0.01:
        left_ear = np.array([int(output_coco[0, 16, 0, 0]*w), int(output_coco[0, 16, 0, 1]*h)])
        neck_coco = np.array([int(output_coco[0, 1, 0, 0]*w), int(output_coco[0, 1, 0, 1]*h)])
        left_shoulder = np.array([int(output_coco[0, 5, 0, 0]*w), int(output_coco[0, 5, 0, 1]*h)])
        
        # 각도 계산 전 키포인트 위치 확인
        print("Left Ear Coordinates:", left_ear)
        print("Neck Coordinates:", neck_coco)
        print("Left Shoulder Coordinates:", left_shoulder)

        angle_coco = calculate_angle_using_atan2(left_ear, neck_coco, left_shoulder)
        print(f"Direction: {direction}. Angle (COCO using atan2): {angle_coco} degrees")

    # ... (나머지 코드와 같음)

    frame_with_all_keypoints = draw_all_keypoints_on_image(output_coco, frame, w, h)
    cv2.imshow('Real-time Keypoints Visualization', frame_with_all_keypoints)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
