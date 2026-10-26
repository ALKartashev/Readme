import cv2
import time

def main():
    """Основная функция для захвата и отображения видео"""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Не удалось подключить камеру')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print('Камера подлючена')

    start_time = time.time()
    frame_count = 0 
    fps = 0

    while True:
        ret, frame = cap.read()

        frame_count += 1
        if time.time() - start_time >= 1:
            fps =  frame_count
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f'FPS : {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not ret:
            print('! Ошибка: не удалось получить видео с камеры')
            break
        cv2.imshow('USB camera', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print('Выход по запросу')
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Программа завершена')

if __name__ == "__main__":
    main()

