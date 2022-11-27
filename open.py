import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("camera open failed")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Can't read camera")
        break

    cv.imshow('PC_camera', img)
    if cv.waitKey(1) == ord('c'):
        img_captured = cv.imwrite('img_captured.png', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()