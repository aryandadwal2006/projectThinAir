import speech_recognition as sr
import pyttsx3
import threading
import time
import queue

class VoiceRecognizer:
    def __init__(self):
        # Set up the recognizer and TTS engine
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Set up the microphone
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        # Command history
        self.last_command = ""
        self.sketch_description = ""
        
        # Threading setup
        self.command_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.speak_lock = threading.Lock()
        
        # Command keywords
        self.commands = {
            "draw": ["draw", "sketch", "start drawing"],
            "stop": ["stop", "pause", "stop drawing"],
            "clear": ["clear", "erase", "delete", "reset"],
            "describe": ["describe", "this is", "i am drawing", "i'm drawing"],
            "help": ["help", "commands", "what can i say"],
            "quit": ["quit", "exit", "close"]
        }
    
    def start_listening(self):
        """Start listening for voice commands in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop)
            self.thread.daemon = True  # Thread will close when main program exits
            self.thread.start()
            self.speak("Voice recognition activated")
    
    def stop_listening(self):
        """Stop the voice recognition thread"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1)
            print("Voice recognition deactivated")
    
    def _listen_loop(self):
        """Keep listening for voice commands in the background"""
        while self.running:
            try:
                with self.microphone as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    self._process_command(text.lower())
                except sr.UnknownValueError:
                    # Couldn't understand the speech
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    
            except sr.WaitTimeoutError:
                # No speech detected, keep listening
                pass
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)  # Avoid tight loop on repeated errors
    
    def _process_command(self, text):
        """Process recognized text and extract commands"""
        self.last_command = text
        command_found = False
        
        # Check for commands
        for cmd_type, phrases in self.commands.items():
            for phrase in phrases:
                if phrase in text:
                    self.command_queue.put((cmd_type, text))
                    command_found = True
                    break
            if command_found:
                break
        
        # If "describe" command, store the description
        if any(phrase in text for phrase in self.commands["describe"]):
            for phrase in self.commands["describe"]:
                if phrase in text:
                    description = text.split(phrase, 1)[1].strip()
                    if description:
                        self.sketch_description = description
                        print(f"Sketch description: {description}")
                        self.speak(f"Description saved: {description}")
        
        # If no command matched, it might be a description
        if not command_found and text:
            self.sketch_description = text
            print(f"Added to sketch context: {text}")
    
    def get_next_command(self):
        """Get the next command from the queue if available"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def speak(self, text):
        """Convert text to speech with error handling"""
        print(f"System: {text}")
        
        # Use a lock to prevent concurrent calls
        if self.speak_lock.acquire(blocking=False):
            try:
                self.engine.say(text)
                try:
                    self.engine.runAndWait()
                except RuntimeError as e:
                    print(f"TTS Engine busy: {e}")
            finally:
                self.speak_lock.release()
        else:
            print("TTS already speaking, skipped additional message")
    
    def get_sketch_description(self):
        """Return the current sketch description"""
        return self.sketch_description
    
    def clear_sketch_description(self):
        """Clear the current sketch description"""
        self.sketch_description = ""