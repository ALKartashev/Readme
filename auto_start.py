import gpiod
import time
import sys
import threading
import subprocess
import os
import shutil
from pathlib import Path

# Конфигурация
CHIP = "gpiochip4"
GPIO_PIN = 2

chip = gpiod.Chip(CHIP)
line = chip.get_line(GPIO_PIN)
line.request(consumer="lamp-script", type=gpiod.LINE_REQ_DIR_OUT)

i = 0


def force_write_image():
    global i
    """
    Запись образа
    """
    image_path = "/home/fpga/Amir_Wi-Fi_full.img"
    target_device = "/dev/sda"

    # Убедимся, что устройство не смонтировано
    try:
        subprocess.run(
            ["sudo", "umount", target_device + "*"], capture_output=True, timeout=30
        )
    except:
        pass  # Игнорируем ошибки размонтирования

    # Записываем с помощью dd
    try:
        print("Запись образа...")

        cmd = [
            "sudo",
            "dd",
            f"if={image_path}",
            f"of={target_device}",
            "bs=4M",
            "status=progress",
            "conv=fsync",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Выводим прогресс
        for line in process.stdout:
            print(line.strip())

        return_code = process.wait()

        if return_code == 0:
            i = 5
            print("Запись завершена успешно!")
            subprocess.run(["sync"], check=True)
            return True
        else:
            i = 0
            line.set_value(1)
            time.sleep(3)
            print(f"Ошибка записи")
            return False

    except Exception as e:
        i = 0
        line.set_value(1)
        time.sleep(3)
        print(f"Ошибка: {e}")
        return False


def main():
    global i
    i = 1
    force_write_image()


def lead_control():
    global i
    while True:
        if i == 5:
            line.set_value(1)
            time.sleep(1)
            line.set_value(0)
            time.sleep(1)
        elif i == 1:
            line.set_value(1)
            time.sleep(0.1)
            line.set_value(0)
            time.sleep(0.1)
        time.sleep(0.01)


threading.Thread(target=main, args=()).start()
time.sleep(0.01)
threading.Thread(target=lead_control, args=()).start()
