import cv2
import numpy as np
import time

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = r'C:\Users\rladn\Desktop\opencv\pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = r'C:\Users\rladn\Desktop\opencv\pose_iter_160000.caffemodel'

# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 카메라 연결
capture = cv2.VideoCapture(0)

# 5초 타이머 시작
start_time = time.time()
while (time.time() - start_time) < 15:
    pass

# 카메라로부터 프레임 읽기
hasFrame, image = capture.read()

# 목 각도 계산 함수
def calculate_neck_angle(points):
    neck = points[BODY_PARTS["Neck"]]
    right_shoulder = points[BODY_PARTS["RShoulder"]]
    left_shoulder = points[BODY_PARTS["LShoulder"]]
    
    if neck and right_shoulder and left_shoulder:
        # 목의 중심 계산
        neck_center = ((right_shoulder[0] + left_shoulder[0]) // 2, (right_shoulder[1] + left_shoulder[1]) // 2)
        
        # 목의 각도 계산
        angle = np.arctan2(neck[1] - neck_center[1], neck[0] - neck_center[0]) * 180 / np.pi
        
        # 이미지에 목과 어깨 사이의 선 그리기
        cv2.line(image, neck, right_shoulder, (0, 255, 0), 2)
        cv2.line(image, neck, left_shoulder, (0, 255, 0), 2)
        
        return angle
    else:
        return None

# 입력 이미지 전처리
frameWidth = image.shape[1]
frameHeight = image.shape[0]
inputWidth = 368
inputHeight = 368
inputScale = 1.0 / 255
mean = [127.5, 127.5, 127.5]
swapRB = False
crop = False
inpBlob = cv2.dnn.blobFromImage(image, inputScale, (inputWidth, inputHeight), mean, swapRB, crop)

# 네트워크에 입력 설정
net.setInput(inpBlob)

# 네트워크 실행
output = net.forward()

# 키포인트 검출 및 그리기
points = []
for i in range(len(BODY_PARTS)):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = int((frameWidth * point[0]) / output.shape[3])
    y = int((frameHeight * point[1]) / output.shape[2])
    if prob > 0.1:
        cv2.circle(image, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    points.append((x, y))
else:
    points.append(None)

# 목 각도 계산
neck_angle = calculate_neck_angle(points)

# 목 각도를 화면에 표시
if neck_angle is not None:
    cv2.putText(image, "Neck Angle: {:.2f} degrees".format(neck_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 결과 이미지 표시
cv2.imshow("Pose Detection", image)

# 사진 저장
cv2.imwrite("captured_image_with_angle.jpg", image)
print("Captured image with angle saved as 'captured_image_with_angle.jpg'")

# 아무 키를 누를 때까지 대기
cv2.waitKey(0)
cv2.destroyAllWindows()