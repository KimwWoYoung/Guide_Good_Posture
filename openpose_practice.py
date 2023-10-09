import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np

def calculate_angle(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)

    # Ensure that the input values are within the valid range for arccos
    a = max(min(a, 1.0), -1.0)
    b = max(min(b, 1.0), -1.0)
    
    # Calculate the cosine of angle C using the cosine law
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

    # Calculate the angle in radians using the arccos function
    angle_rad = np.arccos(cos_C)

    # Convert radians to degrees
    angle = np.degrees(angle_rad)

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

def draw_keypoints_and_connections(output, frame, w, h, color, direction):
    ear = np.array([int(output[0, 16, 0, 0] * w), int(output[0, 16, 0, 1] * h)])
    neck = np.array([int(output[0, 1, 0, 0] * w), int(output[0, 1, 0, 1] * h)])
    shoulder = np.array([int(output[0, 5, 0, 0] * w), int(output[0, 5, 0, 1] * h)])

    if direction == 'right':
        ear = np.array([int(output[0, 17, 0, 0] * w), int(output[0, 17, 0, 1] * h)])
        shoulder = np.array([int(output[0, 2, 0, 0] * w), int(output[0, 2, 0, 1] * h)])

    points = [ear, neck, shoulder]

    for point in points:
        cv2.circle(frame, tuple(point), 5, color[0], -1)
    cv2.line(frame, tuple(points[0]), tuple(points[1]), color[1], 2)
    cv2.line(frame, tuple(points[1]), tuple(points[2]), color[1], 2)

    return frame

# 모델 파일 경로
protoFile_coco = r"C:\Users\rladn\Downloads\openpose-master\openpose-master\models\pose\coco\pose_deploy_linevec.prototxt"
weightsFile_coco = r"C:\Users\rladn\Desktop\opencv\pose_iter_440000.caffemodel"
net_coco = cv2.dnn.readNetFromCaffe(protoFile_coco, weightsFile_coco)

protoFile_mpi = r"C:\Users\rladn\Downloads\openpose-master\openpose-master\models\pose\mpi\pose_deploy_linevec.prototxt"
weightsFile_mpi = r"C:\Users\rladn\Desktop\opencv\pose_iter_160000.caffemodel"
net_mpi = cv2.dnn.readNetFromCaffe(protoFile_mpi, weightsFile_mpi)

image_path = r"C:\Users\rladn\Downloads\medium-shot-man-with-freckles-side-view.jpg"

# 이미지 불러오기
frame = cv2.imread(image_path)

# 이미지 크기 정규화 및 조정
desired_size = (368, 368)  # 원하는 크기로 조정

# 이미지 크기 정규화
normalized_image = frame / 255.0  # 픽셀 값을 0과 1 사이로 정규화

# 이미지 크기 조정
blob = cv2.resize(normalized_image, desired_size)

# COCO keypoints extraction
net_coco = cv2.dnn.readNetFromCaffe(protoFile_coco, weightsFile_coco)
net_coco.setInput(blob)
output_coco = net_coco.forward()

# MPII keypoints extraction
net_mpi = cv2.dnn.readNetFromCaffe(protoFile_mpi, weightsFile_mpi)
net_mpi.setInput(blob)
output_mpi = net_mpi.forward()

direction = direction_determination(output_coco)

# 이미지 너비와 높이 가져오기
h, w = frame.shape[:2]


if direction == 'left':
    left_ear = np.array([int(output_coco[0, 16, 0, 0]*w), int(output_coco[0, 16, 0, 1]*h)])
    neck_coco = np.array([int(output_coco[0, 1, 0, 0]*w), int(output_coco[0, 1, 0, 1]*h)])
    left_shoulder = np.array([int(output_coco[0, 5, 0, 0]*w), int(output_coco[0, 5, 0, 1]*h)])
    angle_coco = calculate_angle(left_ear, neck_coco, left_shoulder)

    neck_mpi = np.array([int(output_mpi[0, 0, 0, 0]*w), int(output_mpi[0, 0, 0, 1]*h)])
    left_ear_mpi = np.array([int(output_mpi[0, 16, 0, 0]*w), int(output_mpi[0, 16, 0, 1]*h)])
    left_shoulder_mpi = np.array([int(output_mpi[0, 6, 0, 0]*w), int(output_mpi[0, 6, 0, 1]*h)])
    angle_mpi = calculate_angle(left_ear_mpi, neck_mpi, left_shoulder_mpi)

elif direction == 'right':
    right_ear = np.array([int(output_coco[0, 17, 0, 0]*w), int(output_coco[0, 17, 0, 1]*h)])
    neck_coco = np.array([int(output_coco[0, 1, 0, 0]*w), int(output_coco[0, 1, 0, 1]*h)])
    right_shoulder = np.array([int(output_coco[0, 2, 0, 0]*w), int(output_coco[0, 2, 0, 1]*h)])
    angle_coco = calculate_angle(right_ear, neck_coco, right_shoulder)

    neck_mpi = np.array([int(output_mpi[0, 0, 0, 0]*w), int(output_mpi[0, 0, 0, 1]*h)])
    right_ear_mpi = np.array([int(output_mpi[0, 17, 0, 0]*w), int(output_mpi[0, 17, 0, 1]*h)])
    right_shoulder_mpi = np.array([int(output_mpi[0, 3, 0, 0]*w), int(output_mpi[0, 3, 0, 1]*h)])
    angle_mpi = calculate_angle(right_ear_mpi, neck_mpi, right_shoulder_mpi)

elif direction == 'front':
    print("Frontal view detected. Please provide a side view image for accurate angle calculation.")
    exit() 

frame = draw_keypoints_and_connections(output_coco, frame, w, h, [(0, 255, 0), (255, 0, 0)], direction)
frame = draw_keypoints_and_connections(output_mpi, frame, w, h, [(255, 0, 255), (0, 0, 255)], direction)

cv2.imshow('Image Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Detected Direction: {direction}")
print(f"Angle (COCO): {angle_coco} degrees")
print(f"Angle (MPI): {angle_mpi} degrees")


