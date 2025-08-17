import cv2
import mediapipe as mp
import requests
import time

# ===== TELEGRAM BOT CONFIG =====
BOT_TOKEN =  "RAo sahab"  # Your bot token
CHAT_ID =  "RAndom" # Replace with your actual chat ID from getUpdates

# Send text alert
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": message})

# Send photo alert
def send_telegram_photo(image_path, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})

# ===== MEDIAPIPE POSE =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
url="http://192.1.3.6060/video"

cap = cv2.VideoCapture(url)

fall_detected = False
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get keypoints for shoulder and hip
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate average shoulder & hip height
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2

        # Calculate vertical distance
        vertical_distance = abs(avg_shoulder_y - avg_hip_y)

        # Calculate horizontal distance between shoulders
        horizontal_shoulder_distance = abs(left_shoulder.x - right_shoulder.x)

        # Detect fall
        if horizontal_shoulder_distance > vertical_distance:
            cv2.putText(frame, " I Fell Down!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            if not fall_detected and time.time() - last_alert_time > 10:
                image_path = f"fall_{int(time.time())}.jpg"
                cv2.imwrite(image_path, frame)  # Save the image
                send_telegram_message(" ALERT: Fall detected!")
                send_telegram_photo(image_path, "Possible fall detected.")
                last_alert_time = time.time()
                fall_detected = True
        else:
            cv2.putText(frame, " Normal", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            fall_detected = False

    cv2.imshow("Fall Detection - MediaPipe Pose", frame)

    # Exit program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
