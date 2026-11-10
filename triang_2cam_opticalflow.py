import cv2
import numpy as np

# ---- ПАРАМЕТРЫ КАМЕРЫ ----

BASELINE_MM = 85.0            # расстояние между центрами линз, мм
FOCAL_PIX = 2000.0            # фокусное расстояние в пикселях (6 мм / 0.003 мм)
FXB = FOCAL_PIX * BASELINE_MM # константа: 170 000 пиксель*мм
HALF_WIDTH = 1280             # половина ширины общего кадра 2560x720

# Параметры Lucas-Kanade
LK_PARAMS = dict(
    winSize=(31, 31),         # размер окна поиска
    maxLevel=5,               # уровней пирамиды
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Глобальные переменные для мыши
click_point = None            # точка клика (x, y) в координатах левого изображения


def mouse_callback(event, x, y, flags, param):
    """Запоминает точку клика на левом изображении."""
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (float(x), float(y))
        print(f"[Клик] Цель: ({x}, {y})")


def split_stereo_frame(combined):
    """Делит склеенный кадр 2560x720 на левое и правое изображения."""
    if combined.shape[1] != HALF_WIDTH * 2:
        combined = cv2.resize(combined, (HALF_WIDTH * 2, 720))
    left = combined[:, :HALF_WIDTH]
    right = combined[:, HALF_WIDTH:]
    return left, right


def compute_distance(disparity_px):
    """Возвращает расстояние в метрах по диспаритету в пикселях."""
    if disparity_px <= 0:
        return -1.0  # некорректный диспаритет
    return (FXB / disparity_px) / 1000.0  # перевод мм -> метры


def main():
    global click_point

    # Открываем камеру
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Принудительно MJPG для максимальной частоты кадров
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру!")
        return

    print(f"Камера открыта. Разрешение: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Константа FxB: {FXB:.0f} пиксель*мм")
    print("ЛКМ по левой половине — измерить расстояние до этой точки.")
    print("Пробел — сбросить точку. ESC — выход.\n")

    cv2.namedWindow("Stereo Camera | Левая половина = клик для замера", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Stereo Camera | Левая половина = клик для замера", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Кадр не получен.")
            break

        # Делим кадр
        left_img, right_img = split_stereo_frame(frame)

        # Делаем копию для отрисовки
        display = np.hstack([left_img, right_img])
        # Рисуем разделительную линию
        cv2.line(display, (HALF_WIDTH, 0), (HALF_WIDTH, 720), (0, 255, 0), 2)
        cv2.putText(display, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "RIGHT", (HALF_WIDTH + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Если есть точка — вычисляем расстояние
        if click_point is not None:
            x_l, y_l = click_point

            # Подготавливаем точку для LK (numpy массив формы (1, 1, 2))
            p0 = np.array([[[x_l, y_l]]], dtype=np.float32)

            # Пирамидальный Lucas-Kanade: левый кадр -> правый кадр
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                left_img, right_img, p0, None, **LK_PARAMS
            )

            if status[0][0] == 1:  # точка успешно отслежена
                x_r, y_r = p1[0][0]

                # Диспаритет (горизонтальное смещение)
                disparity = x_l - x_r
                distance = compute_distance(disparity)

                # Отрисовка
                pt_left = (int(x_l), int(y_l))
                pt_right = (int(x_r) + HALF_WIDTH, int(y_r))

                # Точка на левом
                cv2.circle(display, pt_left, 8, (0, 0, 255), 2)
                cv2.circle(display, pt_left, 3, (0, 0, 255), -1)
                # Точка на правом
                cv2.circle(display, pt_right, 8, (255, 0, 0), 2)
                cv2.circle(display, pt_right, 3, (255, 0, 0), -1)
                # Линия между ними (визуализация смещения)
                cv2.line(display, pt_left, pt_right, (255, 255, 0), 1)

                # Текст с данными
                info_lines = [
                    f"Disparity: {disparity:.2f} px",
                    f"Distance: {distance:.3f} m" if distance > 0 else "Distance: N/A",
                    f"Left: ({x_l:.0f}, {y_l:.0f})",
                    f"Right: ({x_r:.0f}, {y_r:.0f})"
                ]
                y_offset = 60
                for i, line in enumerate(info_lines):
                    cv2.putText(display, line, (10, y_offset + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            else:
                cv2.putText(display, "Потеряна точка! Поток не найден.",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Stereo Camera | Левая половина = клик для замера", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # Пробел — сброс точки
            click_point = None
            print("[Сброс] Точка сброшена.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
