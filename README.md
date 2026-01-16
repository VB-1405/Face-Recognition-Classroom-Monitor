# Classroom Quiet Monitor 🎥🔇

A classroom monitoring project that enrolls students via webcam and helps maintain a quiet testing environment by detecting “talking” using **microphone audio levels (dB)** or **visual motion**—then sending reminders (optionally spoken out loud with TTS).

This repo contains:
- **Python desktop/CLI app** (OpenCV + optional MediaPipe + audio monitoring + TTS + persistent storage)  
- **React demo UI** (front-end simulation showing enrollment + monitoring flow)

> ⚠️ Ethics note: This type of tool should only be used with clear consent, transparent policies, and strong privacy safeguards.

---

## Features

### Python app
- **Student enrollment** via webcam capture (saves samples to disk)
- **Monitoring mode**:
  - Primary: **audio-based** noise detection using decibel threshold (when a microphone is available)
  - Fallback: **visual motion detection** in detected face regions
- **Text-to-speech alerts**
  - macOS: uses native `say`
  - others: attempts `pyttsx3`
- **Persistent storage**
  - student data saved to `data/students.pkl`
  - alerts logged as JSON in `logs/`
- **Alert cooldown** to avoid spamming the same student repeatedly

### React demo UI
- Enrollment + monitoring workflow UI
- Activity log with auto-dismiss alerts
- Uses simulated face/mouth detection placeholders (meant for demoing UI flow, not production detection)

---

# Classroom Quiet Monitor

A Python application that uses computer vision and microphone input to monitor a classroom for noise and identify talking students.

## Features

*   **Face-Based Student Enrollment:** Easily enroll students using a webcam.
*   **Real-Time Face Recognition:** Uses OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer to identify students.
*   **Audio Level Monitoring:** Listens to the microphone to detect noise levels that exceed a configurable threshold.
*   **Visual and Audio Alerts:** Provides on-screen alerts and spoken messages (with a bit of humor) when talking is detected.
*   **Alert Logging:** Keeps a record of all alerts in a JSON file for later review.
*   **Simple Command-Line Interface:** Easy to use menu for enrolling, training, and monitoring.

## How It Works

### Face Recognition

The application uses OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer. 

1.  **Enrollment:** When a student is enrolled, the application captures and saves multiple images of their face.
2.  **Training:** A model is trained on the images of all enrolled students. This trained model is saved to a `trainer.yml` file.
3.  **Recognition:** During monitoring, the application detects faces in the video stream and uses the trained model to predict the identity of the student.

### Audio Monitoring

The application listens to the microphone and calculates the decibel (dB) level of the ambient sound. If the dB level surpasses a configurable threshold for a set duration, it flags this as a "talking" event.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    python classroom_monitor.py
    ```

2.  **Enroll Students:**
    *   Select option **1. Enroll Student** from the main menu.
    *   Enter the student's name and follow the on-screen instructions to capture their face.

3.  **Train the Model:**
    *   After enrolling one or more students, select option **2. Train Model**.
    *   This is a **required step** after enrolling new students to include them in the face recognition model.

4.  **Start Monitoring:**
    *   Select option **3. Start Monitoring** to begin monitoring the classroom.
    *   Press 'q' to stop the monitoring session.

## Dependencies

This project uses the following Python libraries:

*   `opencv-contrib-python`
*   `numpy`
*   `pyaudio`
*   `pyttsx3`
*   `mediapipe`

## Project structure (suggested)

