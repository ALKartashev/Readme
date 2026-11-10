import cv2
import numpy as np
import time

# ---- ПАРАМЕТРЫ КАМЕРЫ ----

BASELINE_MM = 85.0
FOCAL_LENGTH_PIX = 2000.0

# ---- КОРРЕКЦИЯ ----

CORRECTION_A2 = 0.066095
CORRECTION_A1 = 0.282897
CORRECTION_A0 = 0.062187

def correct_distance(raw_m):
    corrected = CORRECTION_A2 * raw_m**2 + CORRECTION_A1 * raw_m + CORRECTION_A0
    return max(corrected, 0.05)

# ---- ФУНКЦИЯ УСРЕДНЕНИЯ С ОТСЕВОМ ВЫБРОСОВ ----

def get_average_distance(depth_map, center_x, center_y, window_size=5, outlier_ratio=0.3):
    """
    Вычисляет усредненное расстояние в окрестности точки,
    отсеивая выбросы.
    
    Параметры:
    - depth_map: карта глубины
    - center_x, center_y: координаты центра
    - window_size: размер окна (должен быть нечетным)
    - outlier_ratio: доля выбросов для отсеивания (0.1 = 10%)
    
    Возвращает:
    - усредненное расстояние (м)
    - количество пикселей, использованных для усреднения
    """
    half = window_size // 2
    
    # Границы окна
    y1 = max(0, center_y - half)
    y2 = min(depth_map.shape[0], center_y + half + 1)
    x1 = max(0, center_x - half)
    x2 = min(depth_map.shape[1], center_x + half + 1)
    
    # Вырезаем область
    region = depth_map[y1:y2, x1:x2]
    
    # Превращаем в одномерный массив
    values = region.flatten()
    
    # Удаляем нулевые значения (фон/ошибки)
    values = values[values > 0.01]
    
    if len(values) == 0:
        return 0.0, 0
    
    # ---- ОТСЕИВАЕМ ВЫБРОСЫ (удаляем слишком далекие/близкие) ----
    # Сортируем значения
    sorted_values = np.sort(values)
    
    # Удаляем outlier_ratio% самых маленьких и самых больших значений
    remove_count = int(len(sorted_values) * outlier_ratio)
    if remove_count > 0:
        trimmed_values = sorted_values[remove_count:-remove_count]
    else:
        trimmed_values = sorted_values
    
    if len(trimmed_values) == 0:
        return np.mean(values) / 1000.0, len(values)
    
    # Вычисляем среднее
    avg_mm = np.mean(trimmed_values)
    avg_m = avg_mm / 1000.0
    
    return avg_m, len(trimmed_values)

# ---- ОТКРЫТИЕ КАМЕРЫ ----

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, test_frame = cap.read()
if ret:
    h, w = test_frame.shape[:2]
    print(f"Размер кадра: {w}x{h}")
else:
    print("Ошибка захвата кадра!")
    cap.release()
    exit()

# ---- СТЕРЕОАЛГОРИТМ (ОГРАНИЧЕННЫЙ ДИАПАЗОН) ----

window_size = 5
min_disp = 0
max_disp = 110
num_disp = max_disp - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

CROSSHAIR_OFFSET_RATIO = 0.15

# ==== НАСТРОЙКИ УСРЕДНЕНИЯ ====
AVG_WINDOW_SIZE = 5      # 5×5 пикселей (можно 3, 5, 7, 9)
OUTLIER_RATIO = 0.3      # Удаляем 30% самых крайних значений

print("СТЕРЕОКАМЕРА - УСРЕДНЕНИЕ ПО 5×5 ПИКСЕЛЕЙ")
print(f"Окно усреднения: {AVG_WINDOW_SIZE}×{AVG_WINDOW_SIZE}")
print(f"Отсев выбросов: {OUTLIER_RATIO*100:.0f}%")
print("ESC - выход, 's' - сохранить")
print("'+' / '-' - изменить размер окна")


# ---- ОСНОВНОЙ ЦИКЛ ----

frame_count = 0
fps_display = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # FPS
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_display = 30 / elapsed if elapsed > 0 else 0
        start_time = time.time()
    
    # Разделяем кадр
    h, w = frame.shape[:2]
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half:]
    
    # Оттенки серого
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    # SGBM
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1
    disparity[disparity > max_disp - 10] = max_disp - 10
    
    # Глубина
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    mask = disparity > 0.1
    depth_map[mask] = (FOCAL_LENGTH_PIX * BASELINE_MM) / disparity[mask]
    depth_map[depth_map > 20000] = 20000
    depth_map[depth_map < 100] = 100
    
    # ---- РАССТОЯНИЕ В ТОЧКЕ ПРИЦЕЛА ----
    
    h_d, w_d = depth_map.shape
    center_x = w_d // 2 + int(w_d * CROSSHAIR_OFFSET_RATIO)
    center_y = h_d // 2
    center_x = min(max(center_x, 10), w_d - 10)
    center_y = min(max(center_y, 10), h_d - 10)
    
    # Используем усреднение с отсевом выбросов
    avg_distance_m, pixels_used = get_average_distance(
        depth_map, 
        center_x, 
        center_y, 
        window_size=AVG_WINDOW_SIZE,
        outlier_ratio=OUTLIER_RATIO
    )
    
    raw_distance_m = avg_distance_m
    corrected_distance_m = correct_distance(raw_distance_m)
    
    # Получаем диспаратность в центре (для отображения)
    center_disparity = disparity[center_y, center_x]
    
    # ---- ОТОБРАЖЕНИЕ ----

    left_display = left.copy()
    
    # Рисуем квадрат 5×5 вокруг прицела
    half_box = AVG_WINDOW_SIZE // 2
    cv2.rectangle(left_display, 
                  (center_x - half_box, center_y - half_box),
                  (center_x + half_box + 1, center_y + half_box + 1),
                  (0, 255, 255), 1)
    
    # Прицел
    crosshair_size = 20
    cv2.line(left_display, (center_x - crosshair_size, center_y), 
             (center_x + crosshair_size, center_y), (0, 255, 0), 2)
    cv2.line(left_display, (center_x, center_y - crosshair_size), 
             (center_x, center_y + crosshair_size), (0, 255, 0), 2)
    cv2.circle(left_display, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Информация
    overlay = left_display.copy()
    cv2.rectangle(overlay, (5, 5), (380, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, left_display, 0.5, 0, left_display)
    
    cv2.putText(left_display, f"DISTANCE: {corrected_distance_m:.2f} m", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(left_display, f"Raw: {raw_distance_m:.2f} m | Disp: {center_disparity:.1f} px", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(left_display, f"Window: {AVG_WINDOW_SIZE}×{AVG_WINDOW_SIZE} | Pixels: {pixels_used} | FPS: {fps_display:.1f}", 
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(left_display, f"Outlier removal: {OUTLIER_RATIO*100:.0f}%", 
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Distance Measurement', left_display)

    # ---- УПРАВЛЕНИЕ ----

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        cv2.imwrite('measurement.jpg', left_display)
        print(f"Сохранено! Расстояние: {corrected_distance_m:.2f} м (усреднено по {AVG_WINDOW_SIZE}×{AVG_WINDOW_SIZE})")
    elif key == ord('+') or key == ord('='):
        AVG_WINDOW_SIZE = min(AVG_WINDOW_SIZE + 2, 21)
        print(f"Размер окна: {AVG_WINDOW_SIZE}×{AVG_WINDOW_SIZE}")
    elif key == ord('-') or key == ord('_'):
        AVG_WINDOW_SIZE = max(AVG_WINDOW_SIZE - 2, 3)
        print(f"Размер окна: {AVG_WINDOW_SIZE}×{AVG_WINDOW_SIZE}")

cap.release()
cv2.destroyAllWindows()
print("Программа завершена.")
