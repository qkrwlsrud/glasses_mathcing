import os
import argparse
import cv2 as cv

from Face_detection import Face_detection

import matplotlib.pyplot as plt

def plot_imgs(origin_img, landmark_img, glasses_img):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title('origin img')
    plt.imshow(origin_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('landmark img')
    plt.imshow(landmark_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('glasses_img')
    plt.imshow(glasses_img)
    plt.axis('off')

    plt.show()

def run(args):
    """
    1. 이미지 캡쳐
    2. 이미지 저장
    3. 얼굴 클래스 호출
    4. 얼굴 이미지 로드
    5. 얼굴 랜드마크 추출
    6. 안경 스티커 붙이기
    반복
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("camera open failed")
        exit()
    while True:
        ret, img = cap.read()

        if not ret:
            print("Can't read camera")
            break

        cv.imshow("Web_Cam", img)

        if cv.waitKey(1) == ord('c'):
            number_of_imgs = len(os.listdir(args.save_dir))
            img_path = os.path.join(args.save_dir, f"captured_img_{number_of_imgs}.jpg")
            img_captured = cv.imwrite(img_path, img)

            # 얼굴인식 클래스 호출
            face = Face_detection(img_path, args.glasses_path, args.dlib_path)
            # 얼굴 이미지 로드
            img_rgb = face.image_load()
            print('img load done')
            # 얼굴에 랜드마크 추출
            landmark_img = face.face_landmark(img_rgb, BGR2RGB=False, verbose=False)
            print('find landmark')
            # 안경 스티커 붙이기
            face_with_sticker = face.attaching_sticker(img_rgb, verbose=False)

            # 시각화
            plot_imgs(img_rgb, landmark_img, face_with_sticker)


        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='captured_img')
    # parser.add_argument("--img_path", type=str, default='test1.jpg')
    parser.add_argument("--glasses_path", type=str, default='glasses_img/sun3.png')
    parser.add_argument("--dlib_path", type=str, default='shape_predictor_68_face_landmark.dat')

    args = parser.parse_args()

    run(args)