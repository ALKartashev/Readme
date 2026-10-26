import cv2
import time
import argparse
import numpy as np

class USBCamera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.zoom_factor = 1.0  # Коэффициент масштабирования
        self.min_zoom = 0.3     # Минимальный зум
        self.max_zoom = 3.0     # Максимальный зум
        self.zoom_step = 0.2    # Шаг изменения зума
        self.show_center_square = False  # Показывать центральный квадрат
        self.center_square_size = 200    # Размер центрального квадрата
        
        # Список рабочих разрешений (проверенные)
        self.resolutions = [
            (640, 480),   # 0 - стандартное VGA
            (320, 240),   # 1 - маленькое QVGA
            (800, 600)    # 2 - среднее SVGA
        ]
        self.current_res_index = 0  # Текущий индекс разрешения
        
    def initialize_camera(self, width=None, height=None):
        """Инициализация камеры с возможностью указания разрешения"""
        if width is None:
            width = self.width
        if height is None:
            height = self.height
            
        # Закрываем предыдущее подключение если было
        if self.cap:
            self.cap.release()
            time.sleep(0.1)  # Небольшая задержка
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Устанавливаем разрешение
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        
        # Даем камере время на инициализацию
        time.sleep(0.5)
        
        # Проверяем подключение
        if not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {self.camera_index}")
            return False
            
        # Проверяем, что камера действительно работает
        ret, frame = self.cap.read()
        if not ret:
            print(f"Ошибка: Камера {self.camera_index} не возвращает кадры")
            self.cap.release()
            return False
            
        print(f"Камера инициализирована с разрешением: {width}x{height}")
        return True
    
    def change_resolution(self):
        """Безопасное изменение разрешения с переподключением"""
        # Переходим к следующему разрешению
        self.current_res_index = (self.current_res_index + 1) % len(self.resolutions)
        new_width, new_height = self.resolutions[self.current_res_index]
        
        print(f"Попытка установить разрешение: {new_width}x{new_height}")
        
        # Пытаемся переподключиться с новым разрешением
        if self.initialize_camera(new_width, new_height):
            print(f"Разрешение успешно изменено: {new_width}x{new_height}")
            return new_width, new_height
        else:
            print(f"Не удалось установить разрешение {new_width}x{new_height}")
            # Пробуем вернуться к предыдущему разрешению
            self.current_res_index = (self.current_res_index - 1) % len(self.resolutions)
            prev_width, prev_height = self.resolutions[self.current_res_index]
            self.initialize_camera(prev_width, prev_height)
            return prev_width, prev_height
    
    def list_available_cameras(self):
        """Поиск доступных камер"""
        available_cameras = []
        for i in range(5):  # Проверяем первые 5 индексов
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
            time.sleep(0.1)
        return available_cameras
    
    def apply_zoom(self, frame):
        """Применение масштабирования к кадру"""
        if self.zoom_factor == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        
        # Вычисляем новые размеры
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)
        
        # Вычисляем координаты для обрезки (центрировано)
        x1 = max(0, (w - new_w) // 2)
        y1 = max(0, (h - new_h) // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        
        # Обрезаем и масштабируем обратно к исходному размеру
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def get_center_square(self, frame):
        """Получение центрального квадрата изображения"""
        h, w = frame.shape[:2]
        size = min(self.center_square_size, min(h, w))
        
        # Вычисляем координаты центрального квадрата
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        
        # Вырезаем центральный квадрат
        center_square = frame[y1:y2, x1:x2]
        
        # Создаем копию кадра и рисуем красную рамку
        frame_with_border = frame.copy()
        cv2.rectangle(frame_with_border, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return center_square, frame_with_border
    
    def show_camera_feed(self):
        """Показ видео с камеры"""
        print("Запуск камеры...")
        print("Управление:")
        print("  'q' - выход")
        print("  's' - показать/скрыть центральный квадрат")
        print("  'w' - увеличить масштаб")
        print("  'e' - уменьшить масштаб")
        print("  'r' - сбросить масштаб")
        print("  'z' - изменить разрешение")
        print("  'f' - переключить полноэкранный режим")
        print("  'c' - сделать снимок центрального квадрата")
        
        fullscreen = False
        frame_count = 0
        start_time = time.time()
        need_reinit = False
        
        while True:
            if need_reinit:
                # Переподключаемся к камере
                if not self.initialize_camera(self.width, self.height):
                    print("Критическая ошибка: не удалось переподключиться к камере")
                    break
                need_reinit = False
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("Ошибка: Не удалось получить кадр, попытка переподключения...")
                need_reinit = True
                time.sleep(1)
                continue
            
            # Применяем масштабирование
            if self.zoom_factor != 1.0:
                frame = self.apply_zoom(frame)
            
            # Обрабатываем центральный квадрат
            center_display = frame.copy()
            if self.show_center_square:
                center_square, frame_with_border = self.get_center_square(frame)
                center_display = frame_with_border
            
            # Добавляем FPS на кадр
            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
                
                cv2.putText(center_display, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Добавляем информацию о разрешении и зуме
            cv2.putText(center_display, f"Res: {self.width}x{self.height}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(center_display, f"Zoom: {self.zoom_factor:.1f}x", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.show_center_square:
                cv2.putText(center_display, "Center Square: ON", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Показываем доступные разрешения
            cv2.putText(center_display, "Press 'z' to change resolution", (10, self.height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Показываем кадр
            window_name = f"USB Camera ({self.camera_index})"
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            cv2.imshow(window_name, center_display)
            
            # Если показываем центральный квадрат, создаем второе окно
            if self.show_center_square:
                center_square, _ = self.get_center_square(frame)
                cv2.imshow("Center Square", center_square)
            else:
                # Закрываем окно центрального квадрата если оно было открыто
                if cv2.getWindowProperty("Center Square", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Center Square")
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_center_square = not self.show_center_square
                print(f"Центральный квадрат: {'ВКЛ' if self.show_center_square else 'ВЫКЛ'}")
            elif key == ord('w'):
                self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
                print(f"Масштаб: {self.zoom_factor:.1f}x")
            elif key == ord('e'):
                self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
                print(f"Масштаб: {self.zoom_factor:.1f}x")
            elif key == ord('r'):
                self.zoom_factor = 1.0
                print("Масштаб сброшен")
            elif key == ord('z'):
                # Закрываем окна перед изменением разрешения
                cv2.destroyAllWindows()
                time.sleep(0.1)
                
                new_width, new_height = self.change_resolution()
                print(f"Новое разрешение: {new_width}x{new_height}")
                
                # Пересоздаем окно
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                continue
            elif key == ord('f'):
                fullscreen = not fullscreen
            elif key == ord('c'):
                if self.show_center_square:
                    self.save_center_screenshot(frame)
                else:
                    print("Сначала включи показ центрального квадрата (клавиша 's')")
            
        self.cleanup()
    
    def save_center_screenshot(self, frame):
        """Сохранение снимка центрального квадрата"""
        center_square, _ = self.get_center_square(frame)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"center_square_{timestamp}.jpg"
        cv2.imwrite(filename, center_square)
        print(f"Снимок центрального квадрата сохранен: {filename}")
    
    def get_camera_info(self):
        """Получение информации о камере"""
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Камера {self.camera_index}: {width}x{height}, FPS: {fps:.1f}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.cap:
            self.cap.release()
        # Закрываем все окна
        cv2.destroyAllWindows()
        print("Камера отключена")

def main():
    parser = argparse.ArgumentParser(description='USB Camera Viewer with Zoom, Center Square and Resolution Control')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Initial frame width')
    parser.add_argument('--height', type=int, default=480, help='Initial frame height')
    parser.add_argument('--square-size', type=int, default=200, help='Center square size')
    parser.add_argument('--list', action='store_true', help='List available cameras')
    
    args = parser.parse_args()
    
    camera = USBCamera(args.camera, args.width, args.height)
    camera.center_square_size = args.square_size
    
    if args.list:
        cameras = camera.list_available_cameras()
        print(f"Доступные камеры: {cameras}")
        return
    
    if camera.initialize_camera():
        camera.get_camera_info()
        print("\nДоступные разрешения:")
        for i, (w, h) in enumerate(camera.resolutions):
            print(f"  {w}x{h}")
        print("\nНажмите 'z' для переключения между разрешениями")
        camera.show_camera_feed()
    else:
        print("Проверьте подключение камеры и попробуйте другой индекс:")
        cameras = camera.list_available_cameras()
        print(f"Доступные камеры: {cameras}")

if __name__ == "__main__":
    main()
