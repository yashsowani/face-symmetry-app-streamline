import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Symmetric landmark pairs from MediaPipe Face Mesh
SYMMETRIC_LANDMARKS = [
    (33, 263), (133, 362), (159, 386), (145, 374),
    (61, 291), (67, 297), (10, 152), (78, 308),
    (95, 324), (107, 336)
]

class FaceSymmetryProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.score = None
        self.smoothed_score = None
        self.alpha = 0.99  # smoothing factor (0 < alpha <= 1)

    def calc_symmetry_score(self, landmarks, img_w):
        # Midline x-position (using landmark 1, nose tip)
        mid_x = landmarks[1].x * img_w

        diffs = []
        for left_idx, right_idx in SYMMETRIC_LANDMARKS:
            left_x = landmarks[left_idx].x * img_w
            right_x = landmarks[right_idx].x * img_w

            left_dist = abs(left_x - mid_x)
            right_dist = abs(right_x - mid_x)

            diffs.append(abs(left_dist - right_dist))

        face_width = abs(landmarks[263].x * img_w - landmarks[33].x * img_w)
        avg_diff = np.mean(diffs) / face_width  # normalized difference

        symmetry_score = max(0, 100 * (1 - avg_diff))
        return symmetry_score

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape

            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            current_score = self.calc_symmetry_score(face_landmarks, w)

            if self.smoothed_score is None:
                self.smoothed_score = current_score
            else:
                self.smoothed_score = self.alpha * current_score + (1 - self.alpha) * self.smoothed_score

            self.score = round(self.smoothed_score, 2)
            text = f"Symmetry Score: {self.score}"
            color = (255, 0, 255)

        else:
            self.score = None
            self.smoothed_score = None
            text = "Detecting face..."
            color = (0, 0, 255)

        cv2.putText(image, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Face Symmetry Detector")
st.title("🔍 Facial Symmetry Detector")

webrtc_streamer(
    key="face-symmetry",
    video_processor_factory=FaceSymmetryProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
