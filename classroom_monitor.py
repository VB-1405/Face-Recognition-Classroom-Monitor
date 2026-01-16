import cv2
import numpy as np
import json
import os
from datetime import datetime
import pickle
import subprocess
import threading
import platform
import pyaudio
import struct
import math
import random

# Try to import mediapipe with compatibility handling
try:
    import mediapipe as mp
    # Check if solutions is available
    if hasattr(mp, 'solutions'):
        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
    else:
        # Newer MediaPipe API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        mp_face_mesh = None
        mp_face_detection = None
        print("Note: Using basic OpenCV face detection (MediaPipe API changed)")
except ImportError:
    print("MediaPipe not available, using OpenCV Haar Cascades")
    mp_face_mesh = None
    mp_face_detection = None

class ClassroomMonitor:
    def __init__(self):
        # Sarcastic/fun messages for 6th graders
        self.alert_messages = [
            "{name}, I can hear you from here. Zip it!",
            "{name}, your voice is lovely, but save it for recess.",
            "{name}, shhh! Even the walls are complaining.",
            "{name}, this is a library moment. Act accordingly.",
            "{name}, I know you have things to say, but not right now buddy.",
            "{name}, the test won't answer itself while you're chatting.",
            "{name}, quiet mode activated... or should I say, YOU should activate it.",
            "{name}, talking during tests? Bold move. Not a smart one though.",
            "{name}, save the commentary for later. Focus time!",
            "{name}, zip it, lock it, put it in your pocket!",
        ]
        
        # Initialize text-to-speech engine
        self.is_macos = platform.system() == 'Darwin'
        
        if self.is_macos:
            # Test if macOS 'say' command works
            try:
                subprocess.run(['say', '-v', '?'], capture_output=True, check=True)
                self.tts_available = True
                print("✓ Text-to-speech initialized (macOS 'say' command)")
            except:
                self.tts_available = False
                print("⚠ Text-to-speech unavailable")
        else:
            # Try pyttsx3 for other platforms
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                self.tts_available = True
                print("✓ Text-to-speech initialized (pyttsx3)")
            except Exception as e:
                self.tts_available = False
                print(f"⚠ Text-to-speech unavailable: {e}")
        
        # Initialize audio monitoring
        self.audio_available = False
        self.audio_stream = None
        self.pyaudio_instance = None
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            # Audio settings
            self.CHUNK = 1024
            self.FORMAT = pyaudio.paInt16
            self.CHANNELS = 1
            self.RATE = 44100
            
            # Test if we can open audio stream
            test_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            test_stream.close()
            
            self.audio_available = True
            print("✓ Audio monitoring initialized (microphone detected)")
        except Exception as e:
            print(f"⚠ Audio monitoring unavailable: {e}")
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
        
        # Decibel threshold (adjust based on your classroom)
        self.decibel_threshold = 60  # Typical conversation is 60-70 dB
        self.noise_detection_duration = 1.5  # Seconds of loud noise before alerting
        self.noise_start_time = None
        
        # Initialize face detection with OpenCV (more compatible)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}
        
        # Data storage
        self.data_dir = "data"
        self.faces_dir = os.path.join(self.data_dir, "faces")
        self.encodings_file = os.path.join(self.data_dir, "students.pkl")
        self.trainer_file = os.path.join(self.data_dir, "trainer.yml")
        self.logs_dir = "logs"
        
        # Create directories if they don't exist
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Load existing students
        self.students = self.load_students()
        self.load_trained_model()
        
        # Mouth detection parameters (for MediaPipe)
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Alert tracking
        self.last_alert_time = {}
        self.alert_cooldown = 10  # seconds between alerts for same student
        
        # Audio monitoring thread
        self.monitoring_active = False
        self.current_decibel = 0
    
    def calculate_decibel(self, audio_data):
        """Calculate decibel level from audio data"""
        try:
            # Convert byte data to integers
            count = len(audio_data) / 2
            format_str = "%dh" % count
            shorts = struct.unpack(format_str, audio_data)
            
            # Calculate RMS (Root Mean Square)
            sum_squares = sum(s ** 2 for s in shorts)
            rms = math.sqrt(sum_squares / count)
            
            # Convert to decibels
            if rms > 0:
                decibel = 20 * math.log10(rms)
                # Normalize to typical range (0-100 dB)
                decibel = max(0, min(100, decibel + 50))
            else:
                decibel = 0
                
            return decibel
        except:
            return 0
    
    def monitor_audio(self):
        """Monitor audio levels in background thread"""
        if not self.audio_available:
            return
                
        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            while self.monitoring_active:
                try:
                    data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                    self.current_decibel = self.calculate_decibel(data)
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠ Audio monitoring error: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
        
    def load_trained_model(self):
        """Load the trained face recognition model."""
        if os.path.exists(self.trainer_file):
            self.recognizer.read(self.trainer_file)
            print("✓ Face recognition model loaded.")

    def train_model(self):
        """Train the face recognizer with enrolled student images."""
        print("\nTraining face recognition model...")
        faces = []
        labels = []
        
        for name, student_data in self.students.items():
            student_id = student_data['id']
            student_dir = os.path.join(self.faces_dir, name)
            
            for image_name in os.listdir(student_dir):
                if image_name.startswith("face_"):
                    image_path = os.path.join(student_dir, image_name)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(student_id)
        
        if not faces:
            print("⚠ No faces found to train the model.")
            return

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.write(self.trainer_file)
        
        print(f"✓ Model trained with {len(faces)} images and saved to {self.trainer_file}")
        self.speak_alert("The face recognition model has been successfully trained.")

    def load_students(self):
        """Load enrolled students from file - PERSISTENT STORAGE"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    students = pickle.load(f)
                
                # Update label_to_name mapping
                self.label_to_name = {data['id']: name for name, data in students.items()}

                print(f"✓ Loaded {len(students)} enrolled students from disk")
                return students
            except Exception as e:
                print(f"⚠ Error loading student data: {e}")
                return {}
        return {}
    
    def save_students(self):
        """Save enrolled students to file - PERSISTENT STORAGE"""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.students, f)
            print(f"✓ Student data saved to disk ({len(self.students)} students)")
        except Exception as e:
            print(f"⚠ Error saving student data: {e}")

    
    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate mouth aspect ratio to detect talking"""
        try:
            upper_lip_points = np.array([[landmarks[i].x, landmarks[i].y] for i in self.UPPER_LIP])
            lower_lip_points = np.array([[landmarks[i].x, landmarks[i].y] for i in self.LOWER_LIP])
            
            vertical_dist = np.mean(np.linalg.norm(upper_lip_points - lower_lip_points, axis=1))
            horizontal_dist = np.linalg.norm(
                np.array([landmarks[61].x, landmarks[61].y]) - 
                np.array([landmarks[291].x, landmarks[291].y])
            )
            
            mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
            return mar
        except:
            return 0
    

    
    def enroll_student(self, name):
        """Enroll a new student by capturing face images."""
        if name in self.students:
            print(f"✗ Student '{name}' is already enrolled.")
            self.speak_alert(f"{name} is already enrolled.")
            return

        print(f"\n{'='*50}")
        print(f"ENROLLING STUDENT: {name}")
        print(f"{'='*50}")
        print("Please look directly at the camera...")
        print("Collecting face data... (15 samples)")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frames_collected = 0
        target_frames = 15
        
        student_dir = os.path.join(self.faces_dir, name)
        os.makedirs(student_dir, exist_ok=True)
        
        while frames_collected < target_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Save face image
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(student_dir, f"face_{frames_collected}.jpg"), face_img)
                
                frames_collected += 1
                
                # Display progress
                progress = int((frames_collected / target_frames) * 100)
                cv2.putText(frame, f"Enrolling: {progress}%", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Student Enrollment', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frames_collected < target_frames:
            print(f"\n✗ Enrollment for {name} cancelled.")
            # Clean up created directory if enrollment is not completed
            if os.path.exists(student_dir):
                for file_name in os.listdir(student_dir):
                    os.remove(os.path.join(student_dir, file_name))
                os.rmdir(student_dir)
            return

        # Assign a new unique ID
        new_id = max(self.label_to_name.keys()) + 1 if self.label_to_name else 1
        
        self.students[name] = {
            'id': new_id,
            'name': name,
            'enrolled_date': datetime.now().isoformat(),
            'total_alerts': 0
        }
        self.save_students()
        
        print(f"\n✓ {name} enrolled successfully!")
        print("Please run 'Train Model' from the main menu to include the new student in the recognition system.")
        self.speak_alert(f"Welcome {name}! Please ask the administrator to train the model to complete your enrollment.")
    
    def start_monitoring(self):
        """Start monitoring the classroom"""
        print(f"\n{'='*50}")
        print("STARTING CLASSROOM MONITORING")
        print(f"{'='*50}")
        
        if not os.path.exists(self.trainer_file):
            print("⚠ Face recognition model not trained. Please train the model first.")
            self.speak_alert("The face recognition model has not been trained yet. Please train the model from the main menu.")
            return

        print(f"Enrolled Students: {len(self.students)}")
        for student_name in self.students.keys():
            print(f"  - {student_name}")
        print(f"\nDetection Method: {'Audio (Decibel-based)' if self.audio_available else 'Visual Motion Detection'}")
        if self.audio_available:
            print(f"Decibel Threshold: {self.decibel_threshold} dB")
            print(f"(Normal conversation: 60-70 dB, Whisper: 30 dB)")
        print("Press 'q' to stop monitoring")
        print(f"{'='*50}\n")
        
        # Start audio monitoring in background
        self.monitoring_active = True
        if self.audio_available:
            audio_thread = threading.Thread(target=self.monitor_audio)
            audio_thread.daemon = True
            audio_thread.start()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        session_start = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Check if noise level exceeds threshold
            noise_detected = False
            if self.audio_available and self.current_decibel > self.decibel_threshold:
                if self.noise_start_time is None:
                    self.noise_start_time = datetime.now()
                else:
                    # Check if sustained noise
                    noise_duration = (datetime.now() - self.noise_start_time).total_seconds()
                    if noise_duration >= self.noise_detection_duration:
                        noise_detected = True
            else:
                self.noise_start_time = None
            
            # Process every other frame for performance
            if frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(100, 100)
                )
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    student_name, confidence = self.identify_student(face_roi)
                    
                    is_talking = noise_detected
                    
                    color = (0, 0, 255) if is_talking and student_name != "Unknown" else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{student_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if is_talking and student_name != "Unknown":
                        cv2.putText(frame, "TALKING", (x, y + h + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        self.send_alert(student_name)
            
            # Display monitoring info
            cv2.putText(frame, "MONITORING ACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Students: {len(self.students)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display audio level if available
            if self.audio_available:
                db_color = (0, 0, 255) if self.current_decibel > self.decibel_threshold else (0, 255, 0)
                cv2.putText(frame, f"Audio: {int(self.current_decibel)} dB", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, db_color, 2)
                
                bar_width = int((self.current_decibel / 100) * 200)
                cv2.rectangle(frame, (10, 100), (10 + bar_width, 115), db_color, -1)
                cv2.rectangle(frame, (10, 100), (210, 115), (255, 255, 255), 2)
                
                threshold_x = 10 + int((self.decibel_threshold / 100) * 200)
                cv2.line(frame, (threshold_x, 100), (threshold_x, 115), (0, 255, 255), 2)
            
            cv2.imshow('Classroom Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.monitoring_active = False
        cap.release()
        cv2.destroyAllWindows()
        
        session_duration = (datetime.now() - session_start).total_seconds()
        print(f"\n{'='*50}")
        print("MONITORING SESSION ENDED")
        print(f"Duration: {session_duration:.0f} seconds")
        print(f"{'='*50}\n")
    
    def identify_student(self, face_image):
        """Identify student from a face image using the trained LBPH model."""
        if not self.label_to_name:
            return "Unknown", 0

        label, confidence = self.recognizer.predict(face_image)
        
        # Lower confidence is better
        if confidence < 100:  # Confidence threshold (adjust as needed)
            name = self.label_to_name.get(label, "Unknown")
            return name, confidence
        
        return "Unknown", confidence
    
    def speak_alert(self, message):
        """Speak the alert message in a separate thread"""
        if not self.tts_available:
            return
        
        def speak():
            try:
                if self.is_macos:
                    # Use macOS native 'say' command (much more reliable)
                    # Using Samantha voice which is polite and clear
                    subprocess.run(['say', '-v', 'Samantha', message], check=False)
                else:
                    # Use pyttsx3 for other platforms
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠ Speech error: {e}")
        
        # Run in separate thread so it doesn't block video processing
        speech_thread = threading.Thread(target=speak)
        speech_thread.daemon = True
        speech_thread.start()
    
    def send_alert(self, student_name):
        """Send sarcastic reminder to student - WITH SPEECH"""
        current_time = datetime.now().timestamp()
        
        if student_name in self.last_alert_time:
            if current_time - self.last_alert_time[student_name] < self.alert_cooldown:
                return
        
        self.last_alert_time[student_name] = current_time
        
        if student_name in self.students:
            self.students[student_name]['total_alerts'] += 1
            self.save_students()  # Save immediately after alert
        
        # Pick a random sarcastic message
        message_template = random.choice(self.alert_messages)
        alert_message = message_template.format(name=student_name)
        
        # Print to console
        print(f"\n🔔 ALERT: {alert_message}")
        
        # SPEAK the alert
        self.speak_alert(alert_message)
        
        # Log to file for permanent record
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'student': student_name,
            'type': 'talking_detected',
            'message': alert_message
        }
        
        log_file = os.path.join(self.logs_dir, f"alerts_{datetime.now().date()}.json")
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        try:
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving log: {e}")
    
    def list_students(self):
        """Display all enrolled students"""
        print(f"\n{'='*50}")
        print("ENROLLED STUDENTS")
        print(f"{'='*50}")
        
        if not self.students:
            print("No students enrolled yet.")
        else:
            for idx, (name, data) in enumerate(self.students.items(), 1):
                enrolled_date = datetime.fromisoformat(data['enrolled_date'])
                print(f"{idx}. {name}")
                print(f"   Enrolled: {enrolled_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"   Total Alerts: {data.get('total_alerts', 0)}")
        
        print(f"{'='*50}\n")
    
    def delete_student(self, name):
        """Remove a student from the system"""
        if name in self.students:
            del self.students[name]
            self.save_students()  # Save changes to disk immediately
            print(f"✓ {name} removed from system")
            self.speak_alert(f"{name} has been removed from the system.")
        else:
            print(f"✗ Student '{name}' not found")


def main():
    print("\n" + "="*50)
    print("CLASSROOM QUIET MONITOR")
    print("Initializing system...")
    print("="*50)
    
    monitor = ClassroomMonitor()
    
    # Show loaded students on startup
    if monitor.students:
        print(f"\n✓ Found {len(monitor.students)} previously enrolled students:")
        for name in monitor.students.keys():
            print(f"  - {name}")
    
    while True:
        print("\n" + "="*50)
        print("CLASSROOM QUIET MONITOR")
        print("="*50)
        print("1. Enroll Student")
        print("2. Train Model")
        print("3. Start Monitoring")
        print("4. List Students")
        print("5. Delete Student")
        print("6. Adjust Decibel Threshold")
        print("7. Exit")
        print("="*50)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter student name: ").strip()
            if name:
                monitor.enroll_student(name)
            else:
                print("Invalid name")
        
        elif choice == '2':
            monitor.train_model()

        elif choice == '3':
            if not monitor.students:
                print("\n⚠ Please enroll at least one student first!")
                monitor.speak_alert("Please enroll at least one student before starting monitoring.")
            else:
                monitor.start_monitoring()
        
        elif choice == '4':
            monitor.list_students()
        
        elif choice == '5':
            monitor.list_students()
            name = input("Enter student name to delete: ").strip()
            if name:
                monitor.delete_student(name)
        
        elif choice == '6':
            if monitor.audio_available:
                print(f"\nCurrent threshold: {monitor.decibel_threshold} dB")
                print("Suggested values:")
                print("  40 dB - Very quiet (whispering)")
                print("  50 dB - Quiet conversation")
                print("  60 dB - Normal conversation (recommended)")
                print("  70 dB - Loud conversation")
                try:
                    new_threshold = int(input("Enter new threshold (30-90): ").strip())
                    if 30 <= new_threshold <= 90:
                        monitor.decibel_threshold = new_threshold
                        print(f"✓ Threshold set to {new_threshold} dB")
                        monitor.speak_alert(f"Decibel threshold updated to {new_threshold}")
                    else:
                        print("Invalid threshold. Must be between 30 and 90.")
                except:
                    print("Invalid input")
            else:
                print("\n⚠ Audio monitoring not available on this system")
        
        elif choice == '7':
            print("\nGoodbye!")
            monitor.speak_alert("Goodbye! Have a great day.")
            if monitor.pyaudio_instance:
                monitor.pyaudio_instance.terminate()
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
