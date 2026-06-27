import cv2

# Load the video
video_path = "traffic.mp4"   # Change path if needed
cap = cv2.VideoCapture(video_path)

# Check if video is loaded
if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

# Read one frame to get datatype
ret, frame = cap.read()

if ret:
    print("Video Loaded Successfully")
    print("-" * 40)
    print(f"Total Number of Frames : {total_frames}")
    print(f"Frame Width            : {width}")
    print(f"Frame Height           : {height}")
    print(f"Frame Shape            : {frame.shape}")
    print(f"Frame Data Type        : {frame.dtype}")
    print(f"Frames Per Second      : {fps:.2f}")
    print(f"Duration (seconds)     : {duration:.2f}")
else:
    print("Could not read a frame.")

cap.release()