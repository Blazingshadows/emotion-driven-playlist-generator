import cv2
import numpy as np
from fer import FER
import logging
from datetime import datetime, timedelta
from collections import deque
import threading
import time

class EmotionDetector:
    def __init__(self, cooldown_seconds=30):
        self.detector = FER(mtcnn=True)
        self.cap = None
        self.is_running = False
        self.current_emotion = None
        self.last_detection_time = None
        self.cooldown_seconds = cooldown_seconds
        self.emotion_queue = deque(maxlen=5)  # Store last 5 emotions
        self.lock = threading.Lock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Emotion weights (probability of selecting songs for each emotion)
        self.emotion_weights = {
            'happy': 0.3,
            'sad': 0.2,
            'angry': 0.15,
            'neutral': 0.1,
            'fear': 0.05,
            'surprise': 0.1,
            'disgust': 0.1
        }

    def start_camera(self):
        """Start the camera capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            self.is_running = True
            self.logger.info("Camera started successfully")
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            raise

    def stop_camera(self):
        """Stop the camera capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Camera stopped")

    def get_dominant_emotion(self, emotions):
        """Get the dominant emotion from detected emotions."""
        if not emotions:
            return None
        return max(emotions.items(), key=lambda x: x[1])[0]

    def should_process_emotion(self):
        """Check if enough time has passed since last emotion detection."""
        if not self.last_detection_time:
            return True
        time_passed = datetime.now() - self.last_detection_time
        return time_passed.total_seconds() >= self.cooldown_seconds

    def process_frame(self):
        """Process a single frame from the camera."""
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Detect emotions in the frame
        result = self.detector.detect_emotions(frame)
        if result:
            # Get the emotions for the first face detected
            emotions = result[0]['emotions']
            dominant_emotion = self.get_dominant_emotion(emotions)
            
            with self.lock:
                self.emotion_queue.append(dominant_emotion)
            
            # Draw emotion text on frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)

        return frame

    def get_current_emotion(self):
        """Get the current emotion with cooldown check."""
        with self.lock:
            if not self.should_process_emotion():
                return None
            
            if not self.emotion_queue:
                return None
            
            # Get most common emotion from queue
            emotions = list(self.emotion_queue)
            emotion = max(set(emotions), key=emotions.count)
            
            self.last_detection_time = datetime.now()
            return emotion

    def get_weighted_emotion(self):
        """Get a weighted random emotion based on probabilities."""
        emotions = list(self.emotion_weights.keys())
        weights = list(self.emotion_weights.values())
        return np.random.choice(emotions, p=weights)

    def run(self):
        """Main loop for emotion detection."""
        self.start_camera()
        
        while self.is_running:
            frame = self.process_frame()
            if frame is not None:
                cv2.imshow('Emotion Detection', frame)
                
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.stop_camera() 