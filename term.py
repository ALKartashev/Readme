import cv2
import numpy as np
import time


def thermal_vision_working():
    """Тепловизор для Raspberry Pi с камерой"""

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Устанавливаем параметры камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Даем камере время на инициализацию
    time.sleep(2)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    print("Управление: Q - выход, T - цветовая карта, +/- - чувствительность")

    # Настройки тепловизора
    colormaps = [
        cv2.COLORMAP_JET,  # Классическая
        cv2.COLORMAP_HOT,  # Горячая
        cv2.COLORMAP_COOL,  # Холодная
        cv2.COLORMAP_BONE,  # Костная
        cv2.COLORMAP_RAINBOW,  # Радужная
    ]

    colormap_names = ["JET", "HOT", "COOL", "BONE", "RAINBOW"]
    current_colormap = 0
    sensitivity = 1.2

    # Для стабилизации температуры
    temp_history = []
    history_size = 5

    # Для отслеживания состояния изображения
    error_count = 0
    max_error_frames = 10

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            # Проверяем качество изображения
            image_status = "Error"
            if ret and frame is not None:
                # Проверяем, что изображение не пустое и не полностью черное/белое
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)

                # Если изображение не полностью черное (0) или белое (255)
                if 10 < mean_intensity < 245:
                    image_status = "Norm"
                    error_count = 0
                else:
                    error_count += 1
            else:
                error_count += 1

            # Если много ошибок подряд, считаем что изображения нет
            if error_count > max_error_frames:
                image_status = "Error"

            if image_status == "Error":
                # Создаем черный экран с сообщением об ошибке
                height, width = 480, 640
                error_display = np.zeros((height, width, 3), dtype=np.uint8)

                # Красная надпись ERROR по центру
                text = "ERROR"
                font_scale = 3
                thickness = 6

                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2

                cv2.putText(
                    error_display,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    thickness,
                )

                # Добавляем информационную панель
                info_bg = np.zeros((100, width, 3), dtype=np.uint8)
                cv2.putText(
                    info_bg,
                    "RPi5 THERMAL CAMERA - NO SIGNAL",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                combined = np.vstack((error_display, info_bg))
                cv2.imshow("Raspberry Pi 5 - Thermal Vision", combined)

                # Пытаемся переподключиться
                if error_count % 30 == 0:  # Каждые 30 кадров ошибки
                    print("Попытка переподключения к камере...")
                    cap.release()
                    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                    time.sleep(1)

            else:
                # Нормальная обработка изображения
                height, width = frame.shape[:2]

                # Зеркальное отображение
                frame = cv2.flip(frame, 1)

                # Конвертируем в grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Улучшаем контраст
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_enhanced = clahe.apply(gray)

                # Легкое размытие для сглаживания
                gray_blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

                # Применяем чувствительность
                gray_thermal = cv2.convertScaleAbs(gray_blur, alpha=sensitivity, beta=0)

                # Автоматическая регулировка диапазона температур
                current_min, current_max = gray_thermal.min(), gray_thermal.max()
                temp_history.append((current_min, current_max))
                if len(temp_history) > history_size:
                    temp_history.pop(0)

                # Вычисляем средний диапазон
                avg_min = np.mean([t[0] for t in temp_history])
                avg_max = np.mean([t[1] for t in temp_history])

                # Нормализуем к хорошему диапазону
                target_min = max(0, int(avg_min - 10))
                target_max = min(255, int(avg_max + 10))

                gray_normalized = cv2.normalize(
                    gray_thermal, None, target_min, target_max, cv2.NORM_MINMAX
                )

                # Применяем тепловую цветовую карту
                thermal = cv2.applyColorMap(
                    gray_normalized, colormaps[current_colormap]
                )

                # Смешиваем с оригиналом
                blended = cv2.addWeighted(thermal, 0.8, frame, 0.2, 0)

                # Добавляем надпись STATUS
                status_text = "Norm"
                status_color = (0, 255, 0)  # Зеленый
                status_bg_color = (0, 0, 0)  # Черный фон

                # Размер и позиция надписи
                text_size = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0]
                text_x = (width - text_size[0]) // 2
                text_y = 50

                # Фон для текста
                cv2.rectangle(
                    blended,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    status_bg_color,
                    -1,
                )

                # Текст статуса
                cv2.putText(
                    blended,
                    status_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2,
                )

                # Добавляем информационную панель
                info_bg = np.zeros((100, width, 3), dtype=np.uint8)

                cv2.putText(
                    info_bg,
                    f"RPi5 THERMAL CAMERA",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    info_bg,
                    f"Map: {colormap_names[current_colormap]}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    info_bg,
                    f"Sens: {sensitivity:.1f} Range: {target_min}-{target_max}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Комбинируем с основным изображением
                combined = np.vstack((blended, info_bg))

                # Расчет FPS
                frame_count += 1
                if time.time() - start_time >= 1.0:
                    fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()
                    cv2.putText(
                        combined,
                        f"FPS: {fps:.1f}",
                        (width - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Отображаем результат
            cv2.imshow("Raspberry Pi 5 - Thermal Vision", combined)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                current_colormap = (current_colormap + 1) % len(colormaps)
                print(f"Цветовая карта: {colormap_names[current_colormap]}")
            elif key == ord("+"):
                sensitivity = min(3.0, sensitivity + 0.1)
                print(f"Чувствительность: {sensitivity:.1f}")
            elif key == ord("-"):
                sensitivity = max(0.5, sensitivity - 0.1)
                print(f"Чувствительность: {sensitivity:.1f}")

    except KeyboardInterrupt:
        print("\nЗавершение работы...")

    except Exception as e:
        print(f"Ошибка в работе тепловизора: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Камера освобождена")


if __name__ == "__main__":
    thermal_vision_working()
