import cv2  # Библиотека OpenCV для работы с изображениями и видео
import numpy as np  # Библиотека для работы с массивами данных
from fastapi import FastAPI, UploadFile  # FastAPI для создания веб-приложения и работы с загружаемыми файлами
from fastapi.responses import HTMLResponse, StreamingResponse  # Классы для возврата HTML-страниц и потоковой передачи данных
from fastapi.staticfiles import StaticFiles  # Класс для работы со статическими файлами
from ultralytics import YOLO  # Библиотека Ultralytics для использования модели YOLO
import tempfile  # Модуль для работы с временными файлами
import os  # Модуль для работы с операционной системой

# Создание экземпляра приложения FastAPI
app = FastAPI()
# Загрузка модели YOLO из файла
model = YOLO('yolov8n-seg.pt')  # Используем модель YOLOv8 для сегментации

# Монтирование директории static для работы со статическими файлами, такими как изображения
app.mount("/static", StaticFiles(directory="static"), name="static")

# Функция для сегментации кадра с использованием модели YOLO
def segment_frame_yolo(frame):
    results = model(frame)  # Применение модели YOLO для сегментации объектов в кадре
    labels = model.names  # Получение меток классов объектов
    found_objects = False  # Флаг для обозначения наличия найденных объектов

    for result in results:
        for i, mask in enumerate(result.masks.data):
            box = result.boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразование координат в целые числа
            confidence = float(box.conf[0])  # Получение уровня уверенности модели
            class_id = int(box.cls[0])  # Получение ID класса объекта

            # Фильтрация только транспортных средств
            if labels[class_id] in ["car", "truck", "bus", "motorbike", "bicycle", "train", "airplane"]:
                found_objects = True  # Установка флага, что найдены объекты
                class_name = labels[class_id]  # Получение имени класса объекта
                # Рисование прямоугольника вокруг объекта
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Добавление текста с именем класса и уровнем уверенности
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Наложение маски на кадр
                color_mask = np.zeros_like(frame)
                resized_mask = cv2.resize(mask.numpy().astype(np.uint8), (frame.shape[1], frame.shape[0]))
                color_mask[resized_mask.astype(bool)] = [0, 255, 0]  # Используем зеленый цвет для маски
                frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

    return frame, found_objects  # Возвращение обработанного кадра и флага наличия объектов

# Определение маршрута для главной страницы
@app.get("/", response_class=HTMLResponse)
async def upload_page():
    # HTML-контент страницы с формами для загрузки изображений и видео
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Сегментация транспортных средств</title>
    </head>
    <body>
        <h1>Загрузите изображение для сегментации</h1>
        <form action="/process-image" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Загрузить</button>
        </form>
        <h1>Загрузите видео для потоковой обработки</h1>
        <form action="/stream-video" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <button type="submit">Потоковая обработка</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)  # Возвращение HTML-страницы

# Определение маршрута для обработки изображений
@app.post("/process-image")
async def process_image(file: UploadFile):
    image_data = await file.read()  # Чтение содержимого загруженного файла
    np_img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)  # Декодирование изображения из байтового массива

    # Сегментация кадра
    processed_frame, found_objects = segment_frame_yolo(np_img)

    if not found_objects:  # Если объекты не найдены
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Результат сегментации</title>
        </head>
        <body>
            <h1>Результат обработки изображения</h1>
            <h2>Транспортные средства не найдены</h2>
            <a href="/">Назад</a>
        </body>
        </html>
        """)

    # Сохранение изображений
    original_path = "static/original.jpg"  # Путь для сохранения оригинального изображения
    processed_path = "static/processed.jpg"  # Путь для сохранения обработанного изображения
    os.makedirs("static", exist_ok=True)  # Создание директории static, если она не существует
    cv2.imwrite(original_path, np_img)  # Сохранение оригинального изображения
    cv2.imwrite(processed_path, processed_frame)  # Сохранение обработанного изображения

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Результат сегментации</title>
    </head>
    <body>
        <h1>Результат обработки изображения</h1>
        <h2>Обработанное изображение</h2>
        <img src="/static/processed.jpg" alt="Processed Image" style="max-width: 100%;"/>
        <br>
        <a href="/">Назад</a>
    </body>
    </html>
    """)

# Определение маршрута для потоковой обработки видео
@app.post("/stream-video")
async def stream_video(file: UploadFile):
    # Сохранение загруженного видео во временный файл
    input_video_path = tempfile.mktemp(suffix=".mp4")
    with open(input_video_path, "wb") as f:
        f.write(await file.read())

    # Проверка на наличие объектов на первом кадре
    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()

    if not ret:  # Если видео не удалось прочитать
        cap.release()
        os.remove(input_video_path)
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ошибка видео</title>
        </head>
        <body>
            <h1>Ошибка обработки видео</h1>
            <h2>Видео не может быть прочитано или пустое.</h2>
            <a href="/">Назад</a>
        </body>
        </html>
        """)

    # Проверка объектов на первом кадре
    _, frame_found_objects = segment_frame_yolo(frame)
    if not frame_found_objects:  # Если объекты не найдены
        cap.release()
        os.remove(input_video_path)
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Результат сегментации</title>
        </head>
        <body>
            <h1>Результат обработки видео</h1>
            <h2>Транспортные средства не найдены</h2>
            <a href="/">Назад</a>
        </body>
        </html>
        """)

    # Генератор для обработки и потоковой передачи кадров
    def process_video_stream():
        while cap.isOpened():  # Пока видео открыто
            ret, frame = cap.read()
            if not ret:  # Если кадры кончились
                break

            # Обработка кадра
            processed_frame, _ = segment_frame_yolo(frame)

            _, buffer = cv2.imencode('.jpg', processed_frame)  # Кодирование кадра в формат JPEG
            frame_bytes = buffer.tobytes()  # Преобразование в байтовый массив

            # Отправка кадра клиенту
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()  # Освобождение видео
        os.remove(input_video_path)  # Удаление временного файла

    # Возврат потока через StreamingResponse
    return StreamingResponse(process_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
