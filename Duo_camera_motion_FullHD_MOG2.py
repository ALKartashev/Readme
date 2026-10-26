import cv2
import numpy as np
import threading
import time
import os
import fcntl
import mmap
import struct
from v4l2 import *

# FullHD разрешение
h_m = 1080
w_m = 1920
shift_X = 0
shift_Y = 0

# Флаг текущей камеры (True - камера 1, False - камера 2)
current_camera = True

# Настройки чувствительности
SENSITIVITY_SETTINGS = {
    "min_contour_area": 2,
    "max_contour_area": 500,
    "var_threshold": 16,
    "learning_rate": 0.02,
    "morph_kernel": (3, 3),
    "binary_threshold": 140,
}

# Глобальные детекторы движения
fgbg_cam1 = None
fgbg_cam2 = None

# Глобальные буферы для кадров FullHD
frame_cam1 = np.zeros((h_m, w_m, 3), dtype=np.uint8)
frame_cam2 = np.zeros((h_m, w_m, 3), dtype=np.uint8)

# История для стабильности
motion_history_cam1 = []
motion_history_cam2 = []
HISTORY_LENGTH = 5

# Оптимизация: предварительное выделение памяти
empty_frame = np.zeros((h_m, w_m, 3), dtype=np.uint8)

def initialize_detectors():
    """Инициализация детекторов"""
    global fgbg_cam1, fgbg_cam2
    
    fgbg_cam1 = cv2.createBackgroundSubtractorMOG2(
        detectShadows=False,
        varThreshold=SENSITIVITY_SETTINGS["var_threshold"],
        history=100
    )
    fgbg_cam2 = cv2.createBackgroundSubtractorMOG2(
        detectShadows=False,
        varThreshold=SENSITIVITY_SETTINGS["var_threshold"], 
        history=100
    )

def shift_image(img, shift_pixels):
    """Сдвиг изображения"""
    if shift_pixels == 0:
        return img
        
    h, w = img.shape[:2]
    shift_pixels = max(-100, min(100, shift_pixels))
    
    if shift_pixels > 0:
        shifted = np.zeros_like(img)
        shifted[shift_pixels:, :] = img[:h-shift_pixels, :]
    else:
        shifted = np.zeros_like(img)
        shifted[:h+shift_pixels, :] = img[-shift_pixels:, :]
    
    return shifted

# Разрешение для обработки 
small_size = (854, 480)  # 16:9 

def detect_motion_simple(frame, fgbg):
    """Детекция движения"""
    if fgbg is None:
        return False, []
        
    settings = SENSITIVITY_SETTINGS
    
    # Уменьшение разрешения с улучшенной интерполяцией
    small_frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_LINEAR)
    
    # Преобразование в grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, settings["morph_kernel"])
    
    # Фоновое вычитание
    fgmask = fgbg.apply(gray, learningRate=settings["learning_rate"])
    
    # Бинаризация
    _, fgmask_binary = cv2.threshold(fgmask, settings["binary_threshold"], 255, cv2.THRESH_BINARY)
    
    # Морфологические операции
    fgmask_clean = cv2.morphologyEx(fgmask_binary, cv2.MORPH_OPEN, kernel)
    fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_CLOSE, kernel)
    
    # Поиск контуров
    contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        min_area = settings["min_contour_area"]
        max_area = settings["max_contour_area"]
        
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Масштабируем координаты обратно к FullHD
            scale_x = w_m / small_size[0]
            scale_y = h_m / small_size[1]
            x_full = int(x * scale_x)
            y_full = int(y * scale_y)
            w_full = max(2, int(w * scale_x))
            h_full = max(2, int(h * scale_y))
            
            motion_boxes.append((x_full, y_full, w_full, h_full, area))
    
    return len(motion_boxes) > 0, motion_boxes

def is_stable_motion(motion_history, current_boxes):
    """Проверка на стабильность"""
    if not current_boxes:
        return False
    
    # Смотрим последние 3 кадра
    recent_frames = motion_history[-3:]
    if len(recent_frames) < 3:
        return False
    
    # Требуем 2 из 3 кадров с обнаружениями
    frames_with_motion = sum(1 for boxes in recent_frames if boxes)
    return frames_with_motion >= 2

def draw_detections(frame, motion_boxes):
    """Отрисовка обнаружений"""
    if not motion_boxes:
        return frame
        
    output_frame = frame.copy()
    
    for i, (x, y, w, h, area) in enumerate(motion_boxes):
        # Рисуем bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Центральный крест
        center_x = x + w // 2
        center_y = y + h // 2
        cross_size = 4
        cv2.line(output_frame, 
                (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), (0, 255, 0), 2)
        cv2.line(output_frame, 
                (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), (0, 255, 0), 2)
    
    return output_frame

def setup_camera(dev_path):
    """Настройка камеры через v4l2 для FullHD"""
    try:
        fd = os.open(dev_path, os.O_RDWR)
        
        # Устанавливаем формат FullHD
        fmt = v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = 1920
        fmt.fmt.pix.height = 1080
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_UYVY
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        
        try:
            fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
        except Exception:
            pass
        
        # Получаем текущий формат
        fcntl.ioctl(fd, VIDIOC_G_FMT, fmt)

        # Выделяем место под буферы
        req = v4l2_requestbuffers()
        req.count = 8  # Количество буферов
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        
        fcntl.ioctl(fd, VIDIOC_REQBUFS, req)

        # Подготовка буферов
        buffers = []
        
        for i in range(req.count):
            buf = v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = i
            
            fcntl.ioctl(fd, VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset)
            buffers.append(mm)
            fcntl.ioctl(fd, VIDIOC_QBUF, buf)

        # Запуск потока
        buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, VIDIOC_STREAMON, buf_type)
        
        return fd, buffers, buf_type
        
    except Exception:
        return None, None, None

def camera_reader(camera_id, dev_path):
    """Чтение кадров с реальной камеры"""
    global frame_cam1, frame_cam2
    
    fd, buffers, buf_type = setup_camera(dev_path)
    if fd is None:
        return
    
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    
    # Предварительные вычисления для обрезки
    crop_y_start = 540 - h_m//2
    crop_y_end = 540 + h_m//2
    crop_x_start = 960 - w_m//2 + shift_X
    crop_x_end = 960 + w_m//2 + shift_X
    
    # Предварительное создание массивов для избежания аллокаций
    frame_shape = (1080, 1920, 2)
    
    while True:
        try:
            # Получаем кадр
            fcntl.ioctl(fd, VIDIOC_DQBUF, buf)
            fresh_buf_index = buf.index
            fresh_frame = buffers[fresh_buf_index]
            
            # Возвращаем буфер в очередь
            fcntl.ioctl(fd, VIDIOC_QBUF, buf)
            
            try:
                # Преобразование кадра
                frame_array = np.frombuffer(fresh_frame, dtype=np.uint8)
                frame_yuyv = frame_array.reshape(frame_shape)
                
                # Преобразование YUYV to BGR
                bgr = cv2.cvtColor(frame_yuyv, cv2.COLOR_YUV2BGR_UYVY)
                
                # Обрезка
                bgr_cropped = bgr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                bgr_shifted = shift_image(bgr_cropped, shift_Y)
                
                if camera_id == 1:
                    frame_cam1[:] = bgr_shifted
                else:
                    frame_cam2[:] = bgr_shifted
                    
            except Exception:
                pass
            
        except Exception:
            time.sleep(0.001)

def display_single():
    """Отображение"""
    global current_camera, fgbg_cam1, fgbg_cam2, motion_history_cam1, motion_history_cam2
    
    initialize_detectors()
    
    window_name = "Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Предварительное создание буферов
    display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Переменные для оптимизации детекции
    motion_detection_counter = 0
    motion_detection_interval = 2  # Детекция каждый 3-й кадр
    
    last_boxes1 = []
    last_boxes2 = []
    last_stable1 = False
    last_stable2 = False
    
    while True:
        start_time = time.time()
        
        # Проверяем, есть ли кадры с камер
        if np.sum(frame_cam1) == 0 or np.sum(frame_cam2) == 0:
            time.sleep(0.01)
            continue
        
        # Выбор текущего кадра в зависимости от активной камеры
        if current_camera:
            current_frame = frame_cam1.copy()
            camera_name = "Camera 1"
            fgbg = fgbg_cam1
            motion_history = motion_history_cam1
            last_boxes = last_boxes1
            last_stable = last_stable1
        else:
            current_frame = frame_cam2.copy()
            camera_name = "Camera 2" 
            fgbg = fgbg_cam2
            motion_history = motion_history_cam2
            last_boxes = last_boxes2
            last_stable = last_stable2
        
        # Оптимизация: детекция движения не на каждом кадре
        motion_detection_counter += 1
        if motion_detection_counter >= motion_detection_interval:
            motion_detected, boxes = detect_motion_simple(current_frame, fgbg)
            
            # Обновляем историю для текущей камеры
            motion_history.append(boxes)
            if len(motion_history) > HISTORY_LENGTH:
                motion_history.pop(0)
            
            # Проверяем стабильность
            stable_motion = is_stable_motion(motion_history, boxes)
            
            # Сохраняем результаты для следующего кадра
            if current_camera:
                last_boxes1 = boxes
                last_stable1 = stable_motion
            else:
                last_boxes2 = boxes
                last_stable2 = stable_motion
            
            motion_detection_counter = 0
        else:
            # Используем результаты предыдущей детекции
            boxes = last_boxes
            stable_motion = last_stable
        
        # Отрисовка обнаружений
        boxes_to_draw = boxes if stable_motion else []
        frame_with_detections = draw_detections(current_frame, boxes_to_draw)
        
        # Отображаем название камеры
        cv2.putText(frame_with_detections, camera_name, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if stable_motion:
            cv2.putText(frame_with_detections, "MOTION DETECTED", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Инструкция по переключению
        cv2.putText(frame_with_detections, "TAB: Switch camera", (20, h_m - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_detections, "ESC: Exit", (20, h_m - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Быстрое масштабирование для отображения
        cv2.resize(frame_with_detections, (1280, 720), display_frame, interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow(window_name, display_frame)
        
        # Оптимизация: регулируем задержку для стабильного FPS
        processing_time = time.time() - start_time
        delay = max(1, int(30 - processing_time * 1000))  # Целевой FPS ~30
        
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 9:  # TAB - переключение камеры
            current_camera = not current_camera
    
    cv2.destroyAllWindows()

def main():
    global current_camera
    
    # Запуск потоков камер
    thread1 = threading.Thread(target=camera_reader, args=(1, "/dev/video0"), daemon=True)
    thread2 = threading.Thread(target=camera_reader, args=(2, "/dev/video8"), daemon=True)
    
    thread1.start()
    thread2.start()
    
    # Ждем инициализации камер
    time.sleep(2)
    
    # Запускаем отображение
    display_single()

if __name__ == "__main__":
    main()
