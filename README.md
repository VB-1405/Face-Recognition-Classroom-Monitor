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

## Project structure (suggested)

