from ultralytics import YOLO
import cv2

# Initialize YOLOv8 object detector
model = YOLO("./yolov8m_best.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model(frame)

    # 탐지된 객체의 클래스 목록을 얻기 
    detected_classes = set()
    for detection in results.xyxy[0]:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        detected_classes.add(class_name)

    print(detected_classes)

    # class에 따라 화면 넘어가기
    if '?' in detected_classes or '?' in detected_classes:
        print("Detected ???, ")
        # 코드
    else:
        print("다시 측정합니다.")

    # 탐지 결과를 화면에 표시
    results.render()
    cv2.imshow("YOLOv5", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 OpenCV 창 종료
cap.release()
cv2.destroyAllWindows()