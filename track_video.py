import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

# Путь к весам модели — ищем epoch60.pt рядом с этим скриптом.
# Path(__file__) — это путь к самому скрипту, .parent — папка где он лежит.
# Благодаря этому скрипт работает на любом компьютере независимо от того,
# в какой папке он находится.
MODEL_PATH = Path(__file__).parent / "epoch60.pt"

# Словарь для поворота видео.
# Ключ — угол в градусах, значение — константа OpenCV для поворота.
# Нужно когда видео снято в портретной ориентации, а модель обучена на прямых кадрах.
ROTATE_CODES = {
    90:  cv2.ROTATE_90_CLOCKWISE,         # повернуть на 90 по часовой
    180: cv2.ROTATE_180,                   # перевернуть вверх ногами
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,  # повернуть на 90 против часовой
}


def draw_tracks(frame, results):
    """Рисует bounding box и подпись для каждого задетектированного объекта."""

    # results[0].boxes — все боксы на текущем кадре
    boxes = results[0].boxes

    # Если модель ничего не нашла — возвращаем кадр без изменений
    if boxes is None or len(boxes) == 0:
        return frame

    for box in boxes:
        # Координаты бокса в пикселях: x1,y1 — верхний левый угол, x2,y2 — нижний правый
        # .cpu().numpy() — переносим тензор с GPU в обычный numpy массив
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Уверенность модели в детекции (от 0 до 1)
        conf = float(box.conf[0].cpu().numpy())

        # ID трека — уникальный номер объекта который ByteTrack присваивает и
        # сохраняет стабильным на протяжении всего видео.
        # Если трекер ещё не успел назначить ID (первые кадры) — ставим -1
        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

        # Зелёный цвет если ID есть, оранжевый если трекер ещё не назначил ID
        color = (0, 255, 0) if track_id != -1 else (0, 165, 255)

        # Текст подписи: "ID:5 0.87" или "det 0.87" если ID нет
        label = f"ID:{track_id} {conf:.2f}" if track_id != -1 else f"det {conf:.2f}"

        # Рисуем прямоугольник вокруг объекта (толщина 3 пикселя)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Вычисляем размер текста чтобы нарисовать под него закрашенный фон
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

        # Закрашенный прямоугольник над боксом — фон для текста чтобы он был читаем
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)

        # Пишем текст поверх закрашенного фона (чёрный цвет)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return frame


def track(source: str, output: str, conf: float, iou: float, rotate: int):
    """Основная функция: читает видео, запускает трекинг, сохраняет результат."""

    # Загружаем YOLO модель с нашими обученными весами
    model = YOLO(str(MODEL_PATH))

    # Открываем входное видео через OpenCV
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source}")

    # Считываем параметры видео чтобы записать выходной файл с теми же настройками
    fps = cap.get(cv2.CAP_PROP_FPS) or 30          # кадров в секунду
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # ширина кадра
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # высота кадра
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в видео

    # При повороте на 90 или 270 градусов ширина и высота меняются местами
    out_w, out_h = (h, w) if rotate in (90, 270) else (w, h)

    # Создаём объект для записи выходного видео
    # "mp4v" — кодек для .mp4 формата
    out_path = Path(output)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    # Получаем код поворота для OpenCV (None если поворот не нужен)
    rotate_code = ROTATE_CODES.get(rotate)

    frame_idx = 0
    while True:
        # Читаем следующий кадр из видео
        # ret=False означает что видео закончилось
        ret, frame = cap.read()
        if not ret:
            break

        # Поворачиваем кадр если задан параметр --rotate
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # Главная строка — запускаем YOLO детекцию + ByteTrack трекинг.
        # tracker="bytetrack.yaml" — выбираем алгоритм трекинга ByteTrack.
        # persist=True — трекер помнит состояние между кадрами,
        #   именно это делает ID стабильными на всём видео.
        # conf — минимальная уверенность детекции (слабые детекции отбрасываются).
        # iou — порог перекрытия боксов для NMS (убирает дубли).
        # verbose=False — отключаем вывод Ultralytics в консоль на каждый кадр.
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        # Рисуем боксы и ID на кадре
        frame = draw_tracks(frame, results)

        # Записываем готовый кадр в выходное видео
        writer.write(frame)

        # Каждые 30 кадров печатаем прогресс чтобы было видно что скрипт не завис
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            n_det = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"  Кадр {frame_idx}/{total} ({pct:.1f}%) | детекций: {n_det}")

    # Освобождаем ресурсы
    cap.release()
    writer.release()
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    # Настраиваем аргументы командной строки
    parser = argparse.ArgumentParser(description="ByteTrack трекинг ценников")

    # Обязательный аргумент — путь к входному видео
    parser.add_argument("source", help="Путь к входному видео")

    # Необязательные аргументы со значениями по умолчанию
    parser.add_argument(
        "--output", default="tracked_output.mp4", help="Путь к выходному видео"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="Порог IoU для NMS")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Повернуть видео перед инференсом (градусы)")

    args = parser.parse_args()

    track(args.source, args.output, args.conf, args.iou, args.rotate)
