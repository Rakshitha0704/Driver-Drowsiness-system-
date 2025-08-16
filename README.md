# Driver-Drowsiness-system-
A real-time driver drowsiness detection system that uses computer vision techniques to monitor facial landmarks and detect signs of fatigue such as eye closure and yawning.   If drowsiness is detected, the system triggers an alarm sound to alert the driver.

# Driver Drowsiness Detection using OpenCV & Mediapipe

A real-time driver drowsiness detection system that uses computer vision techniques to monitor facial landmarks and detect signs of fatigue such as eye closure and yawning.  
If drowsiness is detected, the system triggers an alarm sound to alert the driver.

# Features
- Uses Mediapipe FaceMesh for accurate face landmark detection.
- Calculates:
  - Eye Aspect Ratio (EAR) → detects prolonged eye closure (blinking/drowsiness).
  - Mouth Aspect Ratio (MAR) → detects yawning.
- Threshold-based alert system: raises alarm if eyes remain closed or mouth remains open for several frames.
- Real-time video feed via webcam with visual overlay.
- Works offline, lightweight, and fast.




