import matplotlib.pyplot as plt
import time
import cv2

def capture_frame_times(cap, fps, num_frames=50):
    """Capture frame read times at a given FPS (with delay)."""
    times = []
    delay = 1.0 / fps
    for i in range(num_frames):
        start = time.perf_counter()
        ret, frame = cap.read()
        end = time.perf_counter()
        if not ret:
            print(f"Frame {i} failed at {fps} FPS.")
            times.append(0)
        else:
            times.append(end - start)
        # Sleep to simulate desired FPS
        sleep_time = delay - (end - start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    return times

# Initialize camera
cap = cv2.VideoCapture(700)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Optional warm-up
for _ in range(10):
    cap.read()

# Capture at different FPS
fps_values = [40, 45, 50, 55, 60]
timing_data = {}

for fps in fps_values:
    print(f"Capturing at {fps} FPS...")
    cap.set(cv2.CAP_PROP_FPS, fps)
    timing_data[fps] = capture_frame_times(cap, fps)

cap.release()

# Plotting
plt.figure(figsize=(10, 6))
for fps, times in timing_data.items():
    print(times)
    plt.plot(times, label=f"{fps} FPS")

plt.axhline(y=1/10, color='gray', linestyle='--', label='Ideal 10 FPS (0.1s)')
plt.axhline(y=1/30, color='gray', linestyle='--', label='Ideal 30 FPS (0.033s)')
plt.axhline(y=1/60, color='gray', linestyle='--', label='Ideal 60 FPS (0.016s)')

plt.title("Frame Capture Times at Different FPS")
plt.xlabel("Frame Index")
plt.ylabel("Time (s)")
plt.ylim(0, 0.03)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cam_timing_fps_comparison.png")
print("Saved plot to 'cam_timing_fps_comparison.png'")
