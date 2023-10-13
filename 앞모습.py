# 어깨가 비대칭적이에요 {오른쪽} {왼쪽} 어깨가 올라가있어요
#  머리가 {뒤로 져처어요} {머리가 앞으로 쏠려있어요}- 정면일떄
# 목이 {왼쪽} {오른쪽} 으로 치우쳐져있어요 

# 앞모습일 때
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
#userImageInput = r"C:\Users\rladn\Downloads\WIN_20231011_14_21_56_Pro.jpg"
userImageInput = r'C:\Users\rladn\Downloads\WIN_20231010_16_16_44_Pro.jpg'
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
#outputImagePathWithNames = r"C:\Users\rladn\Desktop\opencv\output_1_with_names.jpg"
outputImagePathWithNames = r"C:\Users\rladn\Desktop\opencv\output_2_with_names.jpg"
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

# 목 각도
def get_neck_angle_based_on_rotation(keypoints):
    left_ear = keypoints.get("LeftEar")
    right_ear = keypoints.get("RightEar")
    neck = keypoints.get("Neck")
    nose = keypoints.get("Nose")
    mean_ear_coordinates = None if not left_ear or not right_ear else ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
    neck_angle = get_angle(mean_ear_coordinates, neck)
    return neck_angle

angle = get_neck_angle_based_on_rotation(keypoints)
print(f"Neck Angle: {angle} degrees")


# 머리의 위치: 목 위쪽 키포인트와 어깨 라인 사이의 거리를 측정
def get_head_to_shoulder_angle(keypoints):
    left_shoulder = keypoints.get("LeftShoulder")
    right_shoulder = keypoints.get("RightShoulder")
    neck = keypoints.get("Neck")
    nose = keypoints.get("Nose")

    if not left_shoulder or not right_shoulder or not neck or not nose:
        return None

    shoulder_midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    
    return get_angle(nose, neck, shoulder_midpoint)

angle = get_head_to_shoulder_angle(keypoints)
print(f"Head to Shoulder Angle: {angle} degrees")

# 얼굴을 수평으로 잘 들고 있는가
def get_ear_angle(keypoints):
    left_ear = keypoints.get("LeftEar")
    right_ear = keypoints.get("RightEar")
    if not left_ear or not right_ear:
        return None
    else:
        face_angle =get_angle(left_ear,right_ear)
    return face_angle
face_angle= get_ear_angle(keypoints)
print(f"face_angle: {face_angle} degrees")

# 왼쪽, 오른쪽 어깨 비대칭성 확인 
def get_shoulder_angles(keypoints):
    left_shoulder = keypoints.get("LeftShoulder")
    right_shoulder = keypoints.get("RightShoulder")
    if not left_shoulder or not right_shoulder:
        return None
    else:
        face_angle =get_angle(left_shoulder,right_shoulder)
    return face_angle
shoulder_angle= get_shoulder_angles(keypoints)
print(f"shoulder_angle: {shoulder_angle} degrees")
#76.06 도이면 왼쪽 어깨가 오른쪽 어깨보다 약간 높아져 있는 것으로 해석됩니다. 즉, 어깨가 왼쪽으로 약간 기울어져 있습니다.
#180 미만 -> 왼쪽이 더 높아요
# 180 초과 -> 오른쪽이 더 높아요
# 180 -> 어깨가 수평이에요


# 머리가 {뒤로 져처어요} {머리가 앞으로 쏠려있어요}- 정면일떄
def get_head_pose_based_on_face(keypoints):
    left_eye = keypoints.get("LeftEye")
    right_eye = keypoints.get("RightEye")
    left_ear = keypoints.get("LeftEar")
    right_ear = keypoints.get("RightEar")

    if not left_eye or not right_eye or not left_ear or not right_ear:
        return "Cannot determine head pose"

    # 왼쪽 눈과 오른쪽 눈의 중심점 계산
    eye_midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # 왼쪽 귀와 오른쪽 귀의 중심점 계산
    ear_midpoint = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)

    # 머리가 뒤로 젖힌 경우
    if ear_midpoint[1] < eye_midpoint[1]:
        return "뒤로 젖힘"

    # 머리가 숙인 경우
    if ear_midpoint[1] > eye_midpoint[1]:
        return "숙임"

    # 머리가 정면에 위치한 경우
    return "정면"

head_position_angle = get_head_pose_based_on_face(keypoints)
print(f"head_position: {head_position_angle} ")

# 올바른자세,I자목,C자목 - 옆모습일때 
# 어깨가 비대칭적이에요 {오른쪽} {왼쪽} 어깨가 올라가있어요
#  머리가 {뒤로 져처어요} {머리가 앞으로 쏠려있어요}- 정면일떄
# 목이 {왼쪽} {오른쪽} 으로 치우쳐져있어요 
# 네가지를 가지고 올바른 자세인지 아닌지 파악



