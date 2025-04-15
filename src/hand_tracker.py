import cv2
import numpy as np
import math

class HandTracker:
    def __init__(self):
        # HSV range for skin (can tweak later)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # For drawing
        self.prev_index_finger = None
        self.is_drawing = False
        self.canvas = None
        
    def detect_hand(self, frame):
        # Init canvas
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)
        
        # BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame, None, self.canvas
        
        # Biggest one assumed as hand
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) < 5000:
            return frame, None, self.canvas
        
        hull = cv2.convexHull(max_contour, returnPoints=False)
        
        try:
            defects = cv2.convexityDefects(max_contour, hull)
        except:
            return frame, None, self.canvas
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if defects is None:
            return frame, None, self.canvas
        
        # Fingertips list
        fingertips = []
        
        # Loop through defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            
            # Distances
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            # Angle
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If angle is sharp, it’s a fingertip
            if angle <= math.pi / 2:
                fingertips.append(end)
                cv2.circle(frame, end, 8, (0, 0, 255), -1)
        
        # Find topmost (index finger prob)
        if fingertips:
            index_finger = min(fingertips, key=lambda p: p[1])
            cv2.circle(frame, index_finger, 12, (255, 0, 0), -1)
            
            if self.is_drawing and self.prev_index_finger:
                cv2.line(self.canvas, self.prev_index_finger, index_finger, (0, 255, 0), 4)
            
            self.prev_index_finger = index_finger
            return frame, index_finger, self.canvas
        
        return frame, None, self.canvas
    
    def toggle_drawing(self):
        self.is_drawing = not self.is_drawing
        return self.is_drawing
    
    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
