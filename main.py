import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from helper_functions import find_polygon_center, save_object, load_object, is_point_in_polygon, get_label_name
from ultralytics import YOLO
import threading

# Load YOLO pretrain model
model = YOLO("Models/yolov8s mAp 45/weights/best.pt")
cv2.setNumThreads(6)
srcVideo = '1'
slot_data = load_object(srcVideo)
points = []
cap = None
frame_canvas = None
running = True
cnt = 1


def get_video_src():
    global cnt
    cnt += 1
    if cnt==4:
        cnt=1
        return str(1)
    return str(cnt % 4)


def on_canvas_click(event):
    global points, slot_data

    x, y = event.x, event.y

    if event.num == 1:  # Left click
        points.append((x, y))
    elif event.num == 3:  # Right click
        for slot in slot_data:
            if is_point_in_polygon((x, y), slot):
                slot_data.remove(slot)
                save_object(slot_data, srcVideo)
                break


def start_video_stream(canvas, info_label):
    global cap, running
    cap = cv2.VideoCapture(f"Media/{srcVideo}.mp4")

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            continue

        frame = cv2.resize(frame, (1280, 720))
        mask_1 = np.zeros_like(frame)
        mask_2 = np.zeros_like(frame)
        results = model(frame, device="cpu")[0]
        slot_data_copy = slot_data.copy()


        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            label_name = get_label_name(class_id)
            if label_name in ["car", "van", "truck"]:
                car_polygon = [
                    (int(x1), int(y1)),
                    (int(x1), int(y2)),
                    (int(x2), int(y2)),
                    (int(x2), int(y1)),
                ]

                for slot in slot_data_copy:
                    polygon_center = find_polygon_center(slot)
                    if is_point_in_polygon(polygon_center, car_polygon):
                        cv2.fillPoly(mask_1, [np.array(slot)], (0, 0, 255))
                        cv2.circle(frame, polygon_center, 2, (255, 0, 255), 6)  # Show center of slot
                        slot_data_copy.remove(slot)


        for slot in slot_data_copy:
            cv2.fillPoly(mask_2, [np.array(slot)], (0, 255, 255))

        frame = cv2.addWeighted(mask_1, 0.2, frame, 1, 0)
        frame = cv2.addWeighted(mask_2, 0.2, frame, 1, 0)


        for idx, slot in enumerate(slot_data):
            center = find_polygon_center(slot)
            slot_id = f"Slot {idx + 1}"  # Add id
            cv2.putText(frame, slot_id, (center[0] - 30, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        total_space = len(slot_data)
        free_space = len(slot_data_copy)

        info_label.config(text=f"Total space: {total_space}\nFree space: {free_space}")


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

        canvas.create_image(0, 0, anchor=tk.NW, image=frame_image)
        canvas.image = frame_image


        for x, y in points:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", outline="red")

        if not running:
            break

    cap.release()


def add_slot():
    if len(points) >= 3:
        slot_data.append(points.copy())
        save_object(slot_data, srcVideo)
        points.clear()


def undo_point():
    if points:
        points.pop()


def stop_video_stream():
    global cap, running
    if cap:
        cap.release()
    running = False


def change_video_source(new_video_src, canvas, info_label):
    global srcVideo, running, slot_data, cap

    stop_video_stream()
    srcVideo = new_video_src

    slot_data.clear()
    slot_data = load_object(srcVideo)

    cap = cv2.VideoCapture(f"Media/{srcVideo}.mp4")


    if not cap.isOpened():
        print(f"Failed to open video source: {srcVideo}")
        return


    running = True
    threading.Thread(target=start_video_stream, args=(canvas, info_label), daemon=True).start()



# Create UI
def create_ui():
    global frame_canvas
    root = tk.Tk()
    root.title("Parking Lot Monitoring System")
    root.geometry("1280x900")
    root.configure(bg="#121212")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "TButton",
        foreground="#FFFFFF",
        font=("Arial", 12),
        padding=10,
        relief="flat"
    )
    style.map(
        "TButton",
        background=[("!active", "#0078D7"), ("active", "#005A9E")],
        foreground=[("!active", "#FFFFFF"), ("active", "#FFD700")]
    )

    frame_canvas = tk.Canvas(root, width=1280, height=720, bg="#000000", highlightthickness=0)
    frame_canvas.pack()

    info_frame = tk.Frame(root, bg="#121212")
    info_frame.pack(pady=10)

    info_label = tk.Label(info_frame, text="Total space: 0\nFree space: 0", font=("Arial", 14), fg="#FFFFFF", bg="#121212")
    info_label.pack()

    frame_canvas.bind("<Button-1>", on_canvas_click)
    frame_canvas.bind("<Button-3>", on_canvas_click)

    controls_frame = tk.Frame(root, bg="#121212")
    controls_frame.pack(pady=20)

    add_slot_button = ttk.Button(controls_frame, text="Add Slot", command=add_slot)
    add_slot_button.grid(row=0, column=0, padx=10)

    undo_button = ttk.Button(controls_frame, text="Undo Point", command=undo_point)
    undo_button.grid(row=0, column=1, padx=10)

    quit_button = ttk.Button(controls_frame, text="Quit", command=lambda: [stop_video_stream(), root.quit()])
    quit_button.grid(row=0, column=2, padx=10)

    change_video_button = ttk.Button(controls_frame, text="Change Video Source", command=lambda: change_video_source(get_video_src(), frame_canvas, info_label))
    change_video_button.grid(row=0, column=3, padx=10)

    threading.Thread(target=start_video_stream, args=(frame_canvas, info_label), daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    create_ui()
