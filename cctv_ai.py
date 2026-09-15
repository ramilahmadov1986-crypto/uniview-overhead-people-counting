import os

# FFmpeg mesajlarini azalt
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import time
import threading
import csv
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# CAMERA
# ============================================================

RTSP_URL = "rtsp://admin:PAROL@IP:554/media/video1"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.30

# Daha sürətli AI
AI_SIZE = 416

# Xətt ekranın ortası
LINE_POSITION = 0.50

# Xəttin ətrafındakı zona
LINE_TOLERANCE = 5

# ============================================================
# GLOBAL
# ============================================================

latest_frame = None

frame_lock = threading.Lock()

running = True


# ============================================================
# TRACKING DATA
# ============================================================

# ID -> son mövqeyi
previous_positions = {}

# ID -> son təsdiqlənmiş tərəf
person_sides = {}

# ID -> son sayılma vaxtı
last_count_time = {}

# Eyni ID üçün minimum sayma intervalı
COUNT_COOLDOWN = 0.8


# ============================================================
# COUNTERS
# ============================================================

up_count = 0

down_count = 0


# ============================================================
# REPORT
# ============================================================

REPORT_FOLDER = "reports"

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

current_date = datetime.now().strftime(
    "%Y-%m-%d"
)

report_file = os.path.join(
    REPORT_FOLDER,
    current_date + "_people_count.csv"
)


# ============================================================
# CREATE REPORT
# ============================================================

def create_report_file():

    if not os.path.exists(report_file):

        with open(
            report_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "Person_ID",
                "Direction"
            ])


create_report_file()


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(
    person_id,
    direction
):

    global current_date
    global report_file

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # Yeni gün
    if today != current_date:

        current_date = today

        report_file = os.path.join(
            REPORT_FOLDER,
            current_date +
            "_people_count.csv"
        )

        create_report_file()

    now = datetime.now()

    with open(
        report_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            today,
            now.strftime("%H:%M:%S"),
            person_id,
            direction
        ])


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread():

    global latest_frame
    global running

    cap = None

    while running:

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        if cap is None or not cap.isOpened():

            print(
                "Connecting to Uniview camera..."
            )

            if cap is not None:

                cap.release()

            cap = cv2.VideoCapture(
                RTSP_URL,
                cv2.CAP_FFMPEG
            )

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            if not cap.isOpened():

                print(
                    "Camera connection failed."
                )

                time.sleep(1)

                continue

            print(
                "Camera connected."
            )

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "Camera frame lost."
            )

            cap.release()

            cap = None

            time.sleep(0.5)

            continue

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        if width > 1280:

            scale = 1280 / width

            frame = cv2.resize(
                frame,
                (
                    1280,
                    int(height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        # ----------------------------------------------------
        # ONLY LATEST FRAME
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = frame


    if cap is not None:

        cap.release()


# ============================================================
# AI THREAD
# ============================================================

def ai_thread():

    global running

    global up_count

    global down_count


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    print(
        "AI model loaded."
    )


    # ========================================================
    # LOOP
    # ========================================================

    while running:

        # ----------------------------------------------------
        # GET LATEST FRAME
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                time.sleep(
                    0.005
                )

                continue

            frame = latest_frame.copy()


        try:

            # =================================================
            # YOLO TRACKING
            # =================================================

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=AI_SIZE,
                tracker="bytetrack.yaml",
                verbose=False
            )


            result = results[0]


            # ------------------------------------------------
            # CHECK
            # ------------------------------------------------

            if result.boxes is None:

                continue

            if result.boxes.id is None:

                continue


            # ------------------------------------------------
            # BOXES
            # ------------------------------------------------

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # IDS
            # ------------------------------------------------

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )


            # ------------------------------------------------
            # LINE
            # ------------------------------------------------

            frame_height = frame.shape[0]

            line_y = int(
                frame_height *
                LINE_POSITION
            )


            # =================================================
            # PERSON LOOP
            # =================================================

            for box, person_id in zip(
                boxes,
                ids
            ):

                person_id = int(
                    person_id
                )


                # ------------------------------------------------
                # BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # =================================================
                # IMPORTANT
                #
                # CAMERA DIRECTLY ABOVE
                #
                # USE CENTER OF PERSON
                # =================================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ------------------------------------------------
                # CURRENT SIDE
                # ------------------------------------------------

                if center_y < (
                    line_y -
                    LINE_TOLERANCE
                ):

                    current_side = "TOP"

                elif center_y > (
                    line_y +
                    LINE_TOLERANCE
                ):

                    current_side = "BOTTOM"

                else:

                    current_side = "LINE"


                # ------------------------------------------------
                # PREVIOUS SIDE
                # ------------------------------------------------

                previous_side = (
                    person_sides.get(
                        person_id
                    )
                )


                # ------------------------------------------------
                # TIME
                # ------------------------------------------------

                now = time.time()

                last_time = (
                    last_count_time.get(
                        person_id,
                        0
                    )
                )


                # =================================================
                # DOWN -> UP
                #
                # AŞAĞIDAN YUXARI
                # ÇIXIŞ
                # =================================================

                if (

                    previous_side == "BOTTOM"

                    and

                    current_side == "TOP"

                    and

                    now - last_time
                    > COUNT_COOLDOWN

                ):

                    down_count += 1


                    last_count_time[
                        person_id
                    ] = now


                    save_event(
                        person_id,
                        "CIXIS"
                    )


                    print(
                        f"ID {person_id} "
                        f"-> ÇIXIŞ +1 "
                        f"(AŞAĞI -> YUXARI)"
                    )


                # =================================================
                # UP -> DOWN
                #
                # YUXARIDAN AŞAĞI
                # GİRİŞ
                # =================================================

                elif (

                    previous_side == "TOP"

                    and

                    current_side == "BOTTOM"

                    and

                    now - last_time
                    > COUNT_COOLDOWN

                ):

                    up_count += 1


                    last_count_time[
                        person_id
                    ] = now


                    save_event(
                        person_id,
                        "GIRIS"
                    )


                    print(
                        f"ID {person_id} "
                        f"-> GİRİŞ +1 "
                        f"(YUXARI -> AŞAĞI)"
                    )


                # ------------------------------------------------
                # SAVE SIDE
                # ------------------------------------------------

                if current_side != "LINE":

                    person_sides[
                        person_id
                    ] = current_side


                # ------------------------------------------------
                # SAVE POSITION
                # ------------------------------------------------

                previous_positions[
                    person_id
                ] = center_y


                # =================================================
                # DRAW PERSON
                # =================================================

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    7,
                    (255, 0, 0),
                    -1
                )


                # ------------------------------------------------
                # ID
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    f"ID {person_id}",
                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        except Exception as e:

            print(
                "AI error:",
                e
            )

            time.sleep(
                0.05
            )


# ============================================================
# START THREADS
# ============================================================

camera = threading.Thread(
    target=camera_thread,
    daemon=True
)

ai = threading.Thread(
    target=ai_thread,
    daemon=True
)

camera.start()

ai.start()


# ============================================================
# START MESSAGE
# ============================================================

print()

print(
    "=========================================="
)

print(
    " UNIVIEW OVERHEAD PEOPLE COUNTING"
)

print(
    "=========================================="
)

print(
    "YUXARI -> AŞAĞI = GİRİŞ"
)

print(
    "AŞAĞI -> YUXARI = ÇIXIŞ"
)

print(
    "BLUE DOT = PERSON CENTER"
)

print(
    "RED LINE = COUNTING LINE"
)

print(
    "Press Q to exit."
)

print()


# ============================================================
# DISPLAY
# ============================================================

while True:

    with frame_lock:

        if latest_frame is None:

            time.sleep(
                0.005
            )

            continue

        frame = latest_frame.copy()


    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    line_y = int(
        height *
        LINE_POSITION
    )


    # --------------------------------------------------------
    # RED LINE
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            0,
            line_y
        ),
        (
            width,
            line_y
        ),
        (0, 0, 255),
        3
    )


    # ========================================================
    # GİRİŞ OXU - YUXARI HİSSƏ
    # ========================================================

    center_x = width // 2

    cv2.arrowedLine(
        frame,
        (
            center_x,
            line_y - 100
        ),
        (
            center_x,
            line_y - 20
        ),
        (0, 255, 0),
        5,
        tipLength=0.35
    )


    # ========================================================
    # ÇIXIŞ OXU - AŞAĞI HİSSƏ
    # ========================================================

    cv2.arrowedLine(
        frame,
        (
            center_x,
            line_y + 100
        ),
        (
            center_x,
            line_y + 20
        ),
        (0, 255, 255),
        5,
        tipLength=0.35
    )


    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "VIRTUAL COUNTING LINE",
        (
            20,
            line_y - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (
            10,
            10
        ),
        (
            350,
            155
        ),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # GİRİŞ
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"GIRIS: {up_count}",
        (
            20,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # ÇIXIŞ
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"CIXIS: {down_count}",
        (
            20,
            80
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


    # --------------------------------------------------------
    # İÇƏRİDƏ QALAN
    # --------------------------------------------------------

    inside_count = (
        up_count -
        down_count
    )


    cv2.putText(
        frame,
        f"ICERIDE QALAN: {inside_count}",
        (
            20,
            115
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    cv2.putText(
        frame,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        (
            20,
            145
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "UNIVIEW OVERHEAD PEOPLE COUNTING",
        frame
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord("q"):

        running = False

        break


# ============================================================
# STOP
# ============================================================

running = False

cv2.destroyAllWindows()


# ============================================================
# FINAL REPORT
# ============================================================

print()

print(
    "=========================================="
)

print(
    " DAILY REPORT"
)

print(
    "=========================================="
)

print(
    f"GİRİŞ (Yuxarı -> Aşağı): {up_count}"
)

print(
    f"ÇIXIŞ (Aşağı -> Yuxarı): {down_count}"
)

print(
    f"İÇƏRİDƏ QALAN: {up_count - down_count}"
)

print()

print(
    "Report:"
)

print(
    report_file
)

print()

print(
    "System stopped."
)