from flask import Flask, render_template, request, Response, jsonify
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
from server import manage_numberplate_db
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
app.latest_detections = []  # Store latest detections

# Initialize PaddleOCR
ocr = PaddleOCR()
model = YOLO("best.pt")

with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

def perform_ocr(image_array):
    if image_array is None:
        return None, 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    result = ocr.ocr(gray, cls=False)
    if result[0]:
        detected_text = [line[1][0] for line in result[0]]
        confidence = float(result[0][0][1][1])
        return ''.join(detected_text), confidence
    return None, 0

@app.route('/get_detections')
def get_detections():
    return jsonify(app.latest_detections)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join('static', 'input.mp4')
            video_file.save(video_path)
            app.latest_detections = []  # Reset detections for new video
            return render_template('index.html', video_uploaded=True)
    return render_template('index.html', video_uploaded=False)

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_video():
    cap = cv2.VideoCapture(os.path.join('static', 'input.mp4'))
    counter = []
    area = [(5, 180), (3, 249), (984, 237), (950, 168)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True, imgsz=240)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                cx = int(x1 + x2) // 2
                cy = int(y1 + y2) // 2

                result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
                if result >= 0:
                    if track_id not in counter:
                        counter.append(track_id)
                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (160, 50))
                        text, ocr_conf = perform_ocr(crop)

                        if text:
                            # Create a semi-transparent background for text
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1, y1-60), (x2, y1), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                            
                            text_str = f"Plate: {text}"
                            conf_str = f"Conf: {ocr_conf:.2f}"
                            cv2.putText(frame, text_str, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, conf_str, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Store detection for display
                            detection = {
                                'plate': text,
                                'confidence': float(ocr_conf),
                                'timestamp': time.strftime('%H:%M:%S')
                            }
                            app.latest_detections.insert(0, detection)
                            # Keep only last 10 detections
                            app.latest_detections = app.latest_detections[:10]

                            # Update database
                            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ')
                            manage_numberplate_db(text)

        mycounter = len(counter)
        cv2.putText(frame, f'{mycounter}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

        # Optimize frame encoding for streaming
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)