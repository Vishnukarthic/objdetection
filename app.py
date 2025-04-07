import streamlit as st
import cv2
import torch
import numpy as np
import smtplib

# Email Alert Function
def send_email_alert(object_detected):
    sender_email = "muhammedismailcr@gmail.com"
    receiver_email = "vishnukarthickk30@gmail.com"
    password = "aqfj oaha wddm qohi"

    subject = "🚨 Train Track Alert!"
    body = f"Warning! A {object_detected} was detected on the train tracks."

    message = f"Subject: {subject}\n\n{body}"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        print("📩 Email Alert Sent!")
    except Exception as e:
        print("Email failed to send:", e)


# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Streamlit UI
st.title("AI Train Track Obstacle Detection")
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Video capture
video = cv2.VideoCapture(0)

def detect_objects(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = results.xyxy[0].cpu().numpy()

    for *box, conf, cls in detections:
        if conf > confidence_threshold:
            object_name = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{object_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 🚨 Send alert if an animal or vehicle is detected
            if object_name in ["person", "dog", "cow", "car"]:
                st.warning(f"⚠️ Alert: {object_name} detected on tracks!")
                send_email_alert(object_name)  # Send Email Alert

    return frame


# Stream video feed
stframe = st.empty()
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    processed_frame = detect_objects(frame)
    stframe.image(processed_frame, channels="BGR")

video.release()
