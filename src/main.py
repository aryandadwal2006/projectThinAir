import cv2
import numpy as np
import time
import threading
from mediapipe_hand_tracker import HandTracker
from voice_recognizer import VoiceRecognizer
from sketch_improver import SketchImprover
from sketch_to_3d import SketchConverter

def main():
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Can't open webcam.")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize stuff
    tracker = HandTracker()
    voice = VoiceRecognizer()
    voice.start_listening()
    improver = SketchImprover()
    converter = SketchConverter()
    
    # State variables
    drawing_active = False
    voice_active = True
    improved_sketch = None
    model_ready = False
    converting = False
    improving = False
    has_improved = False
    
    # Voice commands for 3D conversion and sketch improvement
    voice.commands["convert"] = ["convert", "make 3d", "create model", "convert to 3d"]
    voice.commands["show"] = ["show model", "display model", "visualize", "view model"]
    voice.commands["improve"] = ["improve", "enhance", "make it better", "improve drawing"]
    
    # Create buttons
    def create_button(name, x, y, w, h, color):
        return {"name": name, "x": x, "y": y, "w": w, "h": h, "color": color}
    
    buttons = [
        create_button("Toggle Draw", 10, 420, 120, 40, (0, 120, 255)),
        create_button("Clear", 140, 420, 80, 40, (0, 0, 255)),
        create_button("Improve", 230, 420, 110, 40, (255, 0, 255)),
        create_button("Make 3D", 350, 420, 110, 40, (0, 255, 255)),
        create_button("Show 3D", 470, 420, 110, 40, (0, 255, 0))
    ]
    
    # Instructions
    print("------ SketchTo3D v2.0 ------")
    print("Press 'd' to toggle drawing")
    print("Press 'c' to clear canvas")
    print("Press 'i' to improve sketch")
    print("Press 'm' to convert sketch to 3D model")
    print("Press 's' to show 3D model visualization")
    print("Press 'v' to toggle voice recognition")
    print("Press 'q' to quit")
    print("\nVoice Commands:")
    print("  'draw' or 'start drawing' - Start drawing")
    print("  'stop' or 'stop drawing' - Stop drawing")
    print("  'clear' or 'erase' - Clear canvas")
    print("  'this is...' or 'I am drawing...' - Add description")
    print("  'improve drawing' - Enhance the sketch")
    print("  'convert to 3D' or 'make 3D' - Create 3D model")
    print("  'show model' or 'visualize' - Display 3D model")
    print("  'help' - List commands")
    print("  'quit' or 'exit' - Close application")
    
    def convert_sketch_thread(canvas, description, is_improved=False):
        """Convert sketch to 3D in a separate thread"""
        nonlocal model_ready, converting
        converting = True
        voice.speak("Converting sketch to 3D model...")
        converter.process_sketch(canvas, description, is_improved)
        model_ready = True
        converting = False
        voice.speak("3D model ready")
    
    def improve_sketch_thread(canvas, description):
        """Improve sketch in a separate thread"""
        nonlocal improved_sketch, improving, has_improved
        improving = True
        voice.speak("Improving sketch...")
        improved = improver.improve_sketch(canvas, description)
        improved_sketch = improved
        improving = False
        has_improved = True
        voice.speak("Sketch improved")
    
    def draw_button(frame, button, is_active=False):
        """Draw a button on the frame"""
        cv2.rectangle(frame, (button["x"], button["y"]), 
                     (button["x"] + button["w"], button["y"] + button["h"]), 
                     button["color"], -1)
        
        # Draw border (thicker if active)
        border_thickness = 3 if is_active else 1
        border_color = (255, 255, 255) if is_active else (0, 0, 0)
        cv2.rectangle(frame, (button["x"], button["y"]), 
                     (button["x"] + button["w"], button["y"] + button["h"]), 
                     border_color, border_thickness)
        
        # Add text
        text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = button["x"] + (button["w"] - text_size[0]) // 2
        text_y = button["y"] + (button["h"] + text_size[1]) // 2
        cv2.putText(frame, button["name"], (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def check_button_press(pos, button):
        """Check if a button was pressed"""
        if pos is None:
            return False
        x, y = pos
        return (button["x"] <= x <= button["x"] + button["w"] and 
                button["y"] <= y <= button["y"] + button["h"])
    
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame with hand tracker
        processed_frame, finger_pos, canvas = tracker.detect_hand(frame)
        
        # Draw buttons
        for i, button in enumerate(buttons):
            # Check if button is active
            is_active = False
            if i == 0:  # Toggle Draw button
                is_active = tracker.is_drawing
            elif i == 2:  # Improve button
                is_active = has_improved
            elif i == 3:  # Make 3D button
                is_active = model_ready
                
            draw_button(processed_frame, button, is_active)
            
            # Check if button is pressed
            if finger_pos and not tracker.prev_index_finger:
                if check_button_press(finger_pos, button):
                    if button["name"] == "Toggle Draw":
                        drawing_active = tracker.toggle_drawing()
                        print(f"Drawing: {'ON' if drawing_active else 'OFF'}")
                    elif button["name"] == "Clear":
                        tracker.clear_canvas()
                        print("Canvas cleared")
                        improved_sketch = None
                        has_improved = False
                        model_ready = False
                    elif button["name"] == "Improve":
                        if canvas is not None and np.any(canvas) and not improving:
                            description = voice.get_sketch_description()
                            threading.Thread(
                                target=improve_sketch_thread, 
                                args=(canvas, description)
                            ).start()
                    elif button["name"] == "Make 3D":
                        if not converting:
                            sketch_to_use = improved_sketch if has_improved else canvas
                            is_improved = has_improved
                            
                            if sketch_to_use is not None and np.any(sketch_to_use):
                                description = voice.get_sketch_description()
                                threading.Thread(
                                    target=convert_sketch_thread, 
                                    args=(sketch_to_use, description, is_improved)
                                ).start()
                            else:
                                voice.speak("Nothing to convert. Please draw something first.")
                    elif button["name"] == "Show 3D":
                        if model_ready:
                            print("Displaying 3D model")
                            try:
                               converter.visualize_model()
                            except Exception as e:
                               print(f"Error displaying model: {e}")
                            else:
                               voice.speak("No 3D model available. Convert your sketch first.")
        
        # Process voice commands
        command = voice.get_next_command()
        if command:
            cmd_type, text = command
            if cmd_type == "draw":
                drawing_active = tracker.toggle_drawing()
                if drawing_active:
                    voice.speak("Drawing mode activated")
                
            elif cmd_type == "stop":
                if drawing_active:
                    drawing_active = tracker.toggle_drawing()
                    voice.speak("Drawing paused")
                
            elif cmd_type == "clear":
                tracker.clear_canvas()
                voice.speak("Canvas cleared")
                improved_sketch = None
                has_improved = False
                model_ready = False
            
            elif cmd_type == "improve":
                if canvas is not None and np.any(canvas) and not improving:
                    description = voice.get_sketch_description()
                    threading.Thread(
                        target=improve_sketch_thread, 
                        args=(canvas, description)
                    ).start()
                else:
                    voice.speak("Nothing to improve. Please draw something first.")
            
            elif cmd_type == "convert":
                if not converting:
                    sketch_to_use = improved_sketch if has_improved else canvas
                    is_improved = has_improved
                    
                    if sketch_to_use is not None and np.any(sketch_to_use):
                        description = voice.get_sketch_description()
                        threading.Thread(
                            target=convert_sketch_thread, 
                            args=(sketch_to_use, description, is_improved)
                        ).start()
                    else:
                        voice.speak("Nothing to convert. Please draw something first.")
            
            elif cmd_type == "show":
              if model_ready:
               print("Displaying 3D model")
              try:
               converter.visualize_model()
              except Exception as e:
               print(f"Error displaying model: {e}")
              else:
               voice.speak("No 3D model available. Convert your sketch first.")
                
            elif cmd_type == "help":
                voice.speak("Available commands: draw, stop, clear, improve drawing, convert to 3D, show model, help, and quit")
                
            elif cmd_type == "quit":
                voice.speak("Closing application")
                break
        
        # Get current sketch description
        description = voice.get_sketch_description()
        
        # Determine which canvas to display
        display_canvas = improved_sketch if has_improved else canvas
        
        # Add status texts to frame
        status_bar = np.zeros((50, processed_frame.shape[1], 3), dtype=np.uint8)
        
        # Draw status in status bar
        cv2.putText(status_bar, f"Drawing: {'ON' if tracker.is_drawing else 'OFF'}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(status_bar, f"Voice: {'ON' if voice_active else 'OFF'}", 
                  (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show conversion status
        if model_ready:
            cv2.putText(status_bar, "3D Model: READY", 
                      (370, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif converting:
            cv2.putText(status_bar, "Converting...", 
                      (370, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Show improvement status
        if has_improved:
            cv2.putText(processed_frame, "IMPROVED SKETCH", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        elif improving:
            cv2.putText(processed_frame, "Improving...", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add description if available
        if description:
            # Create a semi-transparent background for text
            text_overlay = np.zeros((40, processed_frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_overlay, f"Description: {description}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Combine status bar and description
            status_bar = np.vstack([status_bar, text_overlay])
        
        # Show the canvas overlaid on the frame
        if display_canvas is not None:
            # Blend the canvas with the original frame
            if has_improved:
                # For improved sketches, we want to show the full color version
                mask = cv2.cvtColor(cv2.cvtColor(display_canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                _, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
                combined = processed_frame * (1 - mask) + display_canvas * mask
            else:
                # For regular sketches, blend with original frame
                combined = cv2.addWeighted(processed_frame, 1, display_canvas, 0.5, 0)
            
            # Combine with status bar
            display_frame = np.vstack([combined, status_bar])
            cv2.imshow('SketchTo3D', display_frame)
        else:
            # Combine with status bar
            display_frame = np.vstack([processed_frame, status_bar])
            cv2.imshow('SketchTo3D', display_frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            drawing_active = tracker.toggle_drawing()
            print(f"Drawing: {'ON' if drawing_active else 'OFF'}")
        elif key == ord('c'):
            tracker.clear_canvas()
            print("Canvas cleared")
            improved_sketch = None
            has_improved = False
            model_ready = False
        elif key == ord('i'):
            # Improve sketch
            if canvas is not None and np.any(canvas) and not improving:
                description = voice.get_sketch_description()
                threading.Thread(
                    target=improve_sketch_thread, 
                    args=(canvas, description)
                ).start()
            else:
                print("Nothing to improve. Please draw something first.")
        elif key == ord('m'):
            # Convert sketch to 3D
            if not converting:
                sketch_to_use = improved_sketch if has_improved else canvas
                is_improved = has_improved
                
                if sketch_to_use is not None and np.any(sketch_to_use):
                    description = voice.get_sketch_description()
                    threading.Thread(
                        target=convert_sketch_thread, 
                        args=(sketch_to_use, description, is_improved)
                    ).start()
                else:
                    print("Nothing to convert. Please draw something first.")
        elif key == ord('s'):
         # Show 3D model
         if model_ready:
          print("Displaying 3D model")
          try:
            converter.visualize_model()
          except Exception as e:
            print(f"Error displaying model: {e}")
         else:
          print("No 3D model available. Convert your sketch first.")
        elif key == ord('v'):
            voice_active = not voice_active
            if voice_active:
                voice.start_listening()
            else:
                voice.stop_listening()
            print(f"Voice recognition: {'ON' if voice_active else 'OFF'}")
        
        # Small delay to reduce CPU usage
        time.sleep(0.01)
    
    # Clean up
    voice.stop_listening()
    time.sleep(0.5)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()