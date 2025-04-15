import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        # Set up MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure the hand tracking model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # 0 is faster, 1 is more accurate
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing state
        self.is_drawing = False
        self.canvas = None
        self.prev_index_finger = None
    
    def detect_hand(self, frame):
        # Make a canvas if it doesn't exist
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        results = self.hands.process(rgb_frame)
        
        # Draw landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get the position of the index finger tip
                index_finger = (
                    int(hand_landmarks.landmark[8].x * frame.shape[1]),
                    int(hand_landmarks.landmark[8].y * frame.shape[0])
                )
                
                # Draw a circle on the index finger
                cv2.circle(frame, index_finger, 10, (255, 0, 0), -1)
                
                # Draw on the canvas if drawing mode is on
                if self.is_drawing and self.prev_index_finger:
                    cv2.line(self.canvas, self.prev_index_finger, index_finger, (0, 255, 0), 4)
                
                self.prev_index_finger = index_finger
                return frame, index_finger, self.canvas
        
        # Reset if no hand is detected
        if not results.multi_hand_landmarks:
            self.prev_index_finger = None
            
        return frame, None, self.canvas
    
    def toggle_drawing(self):
        # Toggle drawing mode
        self.is_drawing = not self.is_drawing
        return self.is_drawing
    
    def clear_canvas(self):
        # Clear the canvas
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)