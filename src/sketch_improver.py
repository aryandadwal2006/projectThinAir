import cv2
import numpy as np
from PIL import Image
import os
from skimage import morphology, feature, color
import matplotlib.pyplot as plt
import time
import random

class SketchImprover:
    def __init__(self):
        self.improved_sketch = None
        self.assets_dir = "../assets"
        self.colors = {
            'house_walls': (245, 222, 179),  # light tan
            'house_roof': (139, 69, 19),     # brown
            'door': (120, 81, 45),           # dark brown
            'window': (173, 216, 230),       # light blue
            'sky': (135, 206, 235),          # sky blue
            'grass': (34, 139, 34),          # forest green
            'path': (210, 180, 140),         # tan
            'tree': (0, 128, 0),             # green
            'water': (65, 105, 225),         # royal blue
            'mountain': (128, 128, 128),     # gray
        }
        
        # Make sure the assets directory exists
        os.makedirs(self.assets_dir, exist_ok=True)
    
    def improve_sketch(self, sketch_image, description=""):
        """
        Clean up and color a rough sketch
        
        Args:
            sketch_image: The drawing canvas
            description: Text description of the sketch
        """
        # Convert to grayscale if needed
        if len(sketch_image.shape) == 3:
            gray_sketch = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_sketch = sketch_image.copy()
        
        # Step 1: Clean up the sketch
        # Threshold to make a binary image
        _, binary = cv2.threshold(gray_sketch, 10, 255, cv2.THRESH_BINARY)
        
        # Invert if lines are dark
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Remove small noise
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=30)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
        cleaned = cleaned.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no big contours, return the original
        if not contours or max(cv2.contourArea(c) for c in contours) < 500:
            return sketch_image
        
        # Step 2: Figure out what the sketch is
        sketch_type = self._analyze_sketch(description, contours)
        
        # Step 3: Color and improve based on type
        improved = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        
        if sketch_type == "house":
            improved = self._improve_house_sketch(contours, binary, improved)
        elif sketch_type == "landscape":
            improved = self._improve_landscape_sketch(contours, binary, improved)
        elif sketch_type == "face":
            improved = self._improve_face_sketch(contours, binary, improved)
        else:  # Default coloring
            improved = self._improve_generic_sketch(contours, binary, improved)
        
        # Final touches
        improved = cv2.GaussianBlur(improved, (3, 3), 0)
        
        # Save the improved sketch
        file_base = "improved_sketch"
        if description:
            # Use part of the description for the filename
            file_base = "_".join(description.lower().split()[:3])
            
        file_path = os.path.join(self.assets_dir, f"{file_base}_improved.png")
        cv2.imwrite(file_path, improved)
        print(f"Improved sketch saved to {file_path}")
        
        # Store the improved sketch
        self.improved_sketch = improved
        
        return improved
    
    def _analyze_sketch(self, description, contours):
        """Guess what the sketch is based on description and contours"""
        # Check description for keywords
        house_keywords = ["house", "home", "building", "cottage", "cabin"]
        landscape_keywords = ["landscape", "mountain", "tree", "forest", "river", "lake"]
        face_keywords = ["face", "person", "man", "woman", "portrait", "head"]
        
        desc_lower = description.lower()
        
        for keyword in house_keywords:
            if keyword in desc_lower:
                return "house"
        
        for keyword in landscape_keywords:
            if keyword in desc_lower:
                return "landscape"
                
        for keyword in face_keywords:
            if keyword in desc_lower:
                return "face"
        
        # Fallback: guess based on contours
        if len(contours) <= 5:
            return "house"
        elif len(contours) > 20:
            return "landscape"
        else:
            return "generic"
    
    def _improve_house_sketch(self, contours, binary, improved):
        """Improve a house sketch"""
        # Find the biggest contour (probably the house)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Fill the house body with a wall color
        wall_mask = np.zeros_like(binary)
        cv2.drawContours(wall_mask, [main_contour], -1, 255, -1)
        improved[wall_mask == 255] = self.colors['house_walls']
        
        # Detect the roof
        roof_height = int(h * 0.3)  # Roof is about 30% of the house height
        roof_mask = np.zeros_like(binary)
        roof_mask[y:y+roof_height, x:x+w] = wall_mask[y:y+roof_height, x:x+w]
        improved[roof_mask == 255] = self.colors['house_roof']
        
        # Find smaller contours for windows or doors
        for contour in contours:
            if contour is not main_contour:
                area = cv2.contourArea(contour)
                if area > 100:  # Ignore tiny contours
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    # Check if contour is inside the house
                    if (x < cx and cx + cw < x + w and 
                        y < cy and cy + ch < y + h):
                        
                        # If it's in the lower part and looks like a door
                        if cy > y + h/2 and ch > cw:
                            door_mask = np.zeros_like(binary)
                            cv2.drawContours(door_mask, [contour], -1, 255, -1)
                            improved[door_mask == 255] = self.colors['door']
                        else:
                            # Assume it's a window
                            window_mask = np.zeros_like(binary)
                            cv2.drawContours(window_mask, [contour], -1, 255, -1)
                            improved[window_mask == 255] = self.colors['window']
        
        # Add a simple background (sky and grass)
        # Sky
        sky_mask = np.zeros_like(binary)
        sky_height = y + roof_height + 20  # Sky ends just below the roof
        sky_mask[:sky_height, :] = 255
        # Only color pixels that aren't already colored
        current_mask = np.any(improved > 0, axis=2)
        sky_mask[current_mask] = 0
        improved[sky_mask == 255] = self.colors['sky']
        
        # Grass
        grass_mask = np.zeros_like(binary)
        grass_mask[sky_height:, :] = 255
        grass_mask[current_mask] = 0
        improved[grass_mask == 255] = self.colors['grass']
        
        # Draw the outline of the house
        cv2.drawContours(improved, [main_contour], -1, (0, 0, 0), 2)
        
        # Draw outlines of windows and doors
        for contour in contours:
            if contour is not main_contour:
                area = cv2.contourArea(contour)
                if area > 100:
                    cv2.drawContours(improved, [contour], -1, (0, 0, 0), 1)
        
        return improved
    
    def _improve_landscape_sketch(self, contours, binary, improved):
        """Improve a landscape sketch"""
        # Start with a sky background
        improved[:] = self.colors['sky']
        
        # Fill the bottom with grass
        grass_height = int(improved.shape[0] * 0.7)  # Bottom 70% is ground
        improved[grass_height:, :] = self.colors['grass']
        
        # Sort contours by y-position (top to bottom)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # Process contours based on position
        for i, contour in enumerate(sorted_contours):
            area = cv2.contourArea(contour)
            if area < 100:
                continue  # Skip tiny contours
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # If near the top, likely mountains
            if y < improved.shape[0] * 0.3:
                mountain_mask = np.zeros_like(binary)
                cv2.drawContours(mountain_mask, [contour], -1, 255, -1)
                improved[mountain_mask == 255] = self.colors['mountain']
            
            # If in the middle, could be trees
            elif y < improved.shape[0] * 0.6:
                tree_mask = np.zeros_like(binary)
                cv2.drawContours(tree_mask, [contour], -1, 255, -1)
                improved[tree_mask == 255] = self.colors['tree']
            
            # If at the bottom, could be water or path
            else:
                if w > h * 2:  # Wide and short, likely water
                    water_mask = np.zeros_like(binary)
                    cv2.drawContours(water_mask, [contour], -1, 255, -1)
                    improved[water_mask == 255] = self.colors['water']
                else:  # Otherwise a path or small object
                    path_mask = np.zeros_like(binary)
                    cv2.drawContours(path_mask, [contour], -1, 255, -1)
                    improved[path_mask == 255] = self.colors['path']
        
        # Draw all contour outlines
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(improved, [contour], -1, (0, 0, 0), 1)
        
        return improved
    
    def _improve_face_sketch(self, contours, binary, improved):
        """Improve a face sketch"""
        # Simple skin tone background for the face
        skin_tone = (219, 196, 164)  # Light skin tone
        
        # Find the biggest contour (probably the face)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Fill the background with a light color
        improved[:] = (240, 240, 240)
        
        # Fill the face with skin tone
        face_mask = np.zeros_like(binary)
        cv2.drawContours(face_mask, [main_contour], -1, 255, -1)
        improved[face_mask == 255] = skin_tone
        
        # Draw all contour outlines
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(improved, [contour], -1, (0, 0, 0), 1)
        
        return improved
    
    def _improve_generic_sketch(self, contours, binary, improved):
        """Generic improvement for any sketch"""
        # Create a light background
        improved[:] = (240, 240, 240)
        
        # Get a list of colors to use
        color_values = list(self.colors.values())
        
        # Fill each significant contour with a different color
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:
                color = color_values[i % len(color_values)]
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                improved[mask == 255] = color
        
        # Draw all contour outlines
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(improved, [contour], -1, (0, 0, 0), 1)
        
        return improved
    
    def get_improved_sketch(self):
        """Return the current improved sketch"""
        return self.improved_sketch