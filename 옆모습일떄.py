import collections
import time
import cv2
import numpy as np
import math 
import collections
import time
import cv2
import numpy as np
import math 

# COCO 모드 설정
protoFile = r"C:\Users\rladn\Desktop\opencv\pose_deploy_linevec.prototxt"
weightsFile = r"C:\Users\rladn\Desktop\opencv\pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 1], [6, 5], [7, 6], [8, 1], [9, 8], [10, 9], [11, 1], [12, 11], [13, 12], 
              [14, 0], [15, 0], [16, 14], [17, 15]]


# 사용자 이미지 및 백그라운드 이미지 로드
#userImageInput = r"C:\Users\rladn\Downloads\yeah.png"
userImageInput = r"C:\Users\rladn\Downloads\wrong.png"
frame = cv2.imread(userImageInput)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# OpenCV의 딥러닝 모듈로 네트워크 로드
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 네트워크 입력 설정
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)

# 네트워크를 통해 예측 수행
output = net.forward()

# 네트워크 출력 크기 확인
H = output.shape[2]
W = output.shape[3]

# 키포인트 좌표 저장할 리스트 초기화
points = []

# 각 키포인트에 대한 처리
for i in range(nPoints):
    # 해당 키포인트의 신뢰도 맵 추출
    probMap = output[0, i, :, :]

    # 신뢰도 맵에서 가장 높은 값을 찾음
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원본 이미지에 맞게 좌표 스케일 조정
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    # 신뢰도가 임계값보다 큰 경우에만 키포인트로 처리
    if prob > 0.1:
        points.append((int(x), int(y)))
    else:
        points.append(None)

# 키포인트를 선으로 연결하고 그림
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)

# 이미지에 키포인트 이름을 추가하는 함수
def add_keypoint_names(image, keypoints):
    for keypoint_name, (x, y) in keypoints.items():
        if x != "Not Detected":
            cv2.putText(image, keypoint_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


# 결과 이미지 저장
outputImagePath = r"C:\Users\rladn\Desktop\opencv\output_1.jpg" #올바른 자세
#outputImagePath = r"C:\Users\rladn\Desktop\opencv\output_2.jpg" #잘못된 자세
cv2.imwrite(outputImagePath, frame)

keypointsMapping = [
    'Nose',
    'Neck',
    'RightShoulder',
    'RightElbow',
    'RightWrist',
    'LeftShoulder',
    'LeftElbow',
    'LeftWrist',
    'RightHip',
    'RightKnee',
    'RightAnkle',
    'LeftHip',
    'LeftKnee',
    'LeftAnkle',
    'RightEye',
    'LeftEye',
    'RightEar',
    'LeftEar'
]


# 키포인트와 좌표를 매핑하는 딕셔너리 초기화
keypoints = {}

# 각 키포인트에 대한 좌표를 딕셔너리에 추가
for i, keypoint_name in enumerate(keypointsMapping):
    if points[i] is not None:
        x, y = points[i]
        keypoints[keypoint_name] = (x, y)
    else:
        keypoints[keypoint_name] = "Not Detected"

# 결과 딕셔너리 출력
print(keypoints)

def add_keypoint_names(image, keypoints):
    for keypoint_name, point in keypoints.items():
        if point != "Not Detected":
            x, y = point
            cv2.putText(image, keypoint_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


# 이미지에 키포인트 이름 추가
add_keypoint_names(frame, keypoints)

# 결과 이미지 저장
outputImagePathWithNames = r"C:\Users\rladn\Desktop\opencv\output_1_with_names.jpg"
#outputImagePathWithNames = r"C:\Users\rladn\Desktop\opencv\output_2_with_names.jpg"
cv2.imwrite(outputImagePathWithNames, frame)


# 수식 py파일 생성 할 것  
def get_distance(point1, point2):
    """두 점 사이의 거리를 계산합니다."""
    if point1 is None or point2 is None:
        return None
    
    # point2가 두 개의 값을 가지는지 확인합니다.
    if isinstance(point2, (tuple, list)) and len(point2) == 2:
        x2, y2 = point2
    else:
        # point2가 두 개의 값을 가지지 않는 경우를 처리합니다.
        # 오류 메시지를 출력하거나 기본값을 사용할 수 있습니다.
        return None  # 또는 다른 조치를 취하십시오.
    
    x1, y1 = point1
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_angle(a, b, c=None):
    if c is None:  # 두 점 사이의 각도를 계산
        delta_x = b[0] - a[0]
        delta_y = b[1] - a[1]
        angle = abs(math.degrees(math.atan2(delta_y, delta_x)))  # 절대값 적용
        return angle

    else:  # 세 점 사이의 각도를 계산
        ba = (a[0]-b[0], a[1]-b[1])  # 벡터 BA
        bc = (c[0]-b[0], c[1]-b[1])  # 벡터 BC

        dot_product = ba[0]*bc[0] + ba[1]*bc[1]  # 내적
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)  # 벡터 BA의 크기
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)  # 벡터 BC의 크기
        
        angle_rad = math.acos(dot_product / (mag_ba * mag_bc))
        angle_deg = math.degrees(angle_rad)
        return angle_deg

def is_rotated_trunk(keypoints):
    # 이전 코드는 그대로 두고 direction 정보만 반환하도록 수정
    left_eye = keypoints.get("LeftEye")
    right_eye = keypoints.get("RightEye")
    left_ear = keypoints.get("LeftEar")
    right_ear = keypoints.get("RightEar")
    left_shoulder = keypoints.get("LeftShoulder")
    right_shoulder = keypoints.get("RightShoulder")

    # 5개 이상의 키포인트가 존재하지 않으면 False 반환
    valid_keypoints = [left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder]

    # "Not Detected"을 None으로 처리하여 키포인트가 감지되지 않은 것으로 취급
    valid_keypoints = [None if k == 'Not Detected' else k for k in valid_keypoints]

    valid_keypoint_count = sum(1 for k in valid_keypoints if k is not None)

    if valid_keypoint_count < 6:
        right_count = sum(1 for k in [right_eye, right_ear, right_shoulder] if k is not None)
        if right_count >= 2:
            return "RIGHT"
        left_count = sum(1 for k in [left_eye, left_ear, left_shoulder] if k is not None)
        if left_count >= 2:
            return "LEFT"

    return "FALSE"

# 키포인트가 회전된 상태인지 확인
rotated_trunk = is_rotated_trunk(keypoints)
print("Is the trunk rotated?", rotated_trunk)

# 옆모습일 때 
def get_side_neck_angle(keypoints, rotation_information):
    left_ear = keypoints.get("LeftEar")
    right_ear = keypoints.get("RightEar")
    left_shoulder = keypoints.get("LeftShoulder")
    right_shoulder = keypoints.get("RightShoulder")

    direction = rotation_information  # 방향 정보 수정

    if direction == "RIGHT":  # 왼쪽 옆모습을 나타냄
        if left_ear and left_shoulder:
            angle = get_angle(left_ear, left_shoulder)
        else:
            return "Cannot compute neck angle without left ear and left shoulder points."
    elif direction == "LEFT":  # 오른쪽 옆모습을 나타냄
        if right_ear and right_shoulder:
            angle = get_angle(right_ear, right_shoulder)
        else:
            return "Cannot compute neck angle without right ear and right shoulder points."
    else:
        return "Invalid direction information."

    if 80 <= angle <= 100:  # 예상되는 정상 범위
        return f"Normal posture with neck angle: {angle} degrees."
    elif angle < 80:
        return f"Possible forward head posture with neck angle: {angle} degrees."
    else:
        return f"Unexpected posture with neck angle: {angle} degrees."

rotation_information = rotated_trunk  # 방향 정보 수정
neck_angle_info = get_side_neck_angle(keypoints, rotation_information)
print(neck_angle_info)

def get_side_back_angle(keypoints, rotation_information):
    left_shoulder = keypoints.get("LeftShoulder")
    right_shoulder = keypoints.get("RightShoulder")
    left_hip = keypoints.get("LeftHip")
    right_hip = keypoints.get("RightHip")

    direction = rotation_information  # 방향 정보 수정

    if direction == "RIGHT":  # 왼쪽 옆모습을 나타냄
        if left_shoulder and left_hip:
            angle = get_angle(left_shoulder, left_hip)
        else:
            return "Cannot compute back angle without left shoulder and left hip points."
    elif direction == "LEFT":  # 오른쪽 옆모습을 나타냄
        if right_shoulder and right_hip:
            angle = get_angle(right_shoulder, right_hip)
        else:
            return "Cannot compute back angle without right shoulder and right hip points."
    else:
        return "Invalid direction information."

    if 80 <= angle <= 100:  # 예상되는 정상 범위
        return f"Normal posture with back angle: {angle} degrees."
    elif angle < 80:
        return f"Possible forward head posture with back angle: {angle} degrees."
    else:
        return f"Unexpected posture with back angle: {angle} degrees."

rotation_information = rotated_trunk  # 방향 정보 수정
back_angle_info = get_side_back_angle(keypoints, rotation_information)
print(back_angle_info)



# 올바른자세,I자목,C자목 - 옆모습일때
# 등 피는지 구부리는지 


