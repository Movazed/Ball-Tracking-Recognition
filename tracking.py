import cv2
import os

def define_quadrants(frame):
    height, width, _ = frame.shape
    # Define the red tape boundaries
    quadrant_width = width // 2
    quadrant_height = height // 2
    
    quadrants = {
        1: (quadrant_width, quadrant_height, width, height),    # Bottom-right
        2: (0, quadrant_height, quadrant_width, height),        # Bottom-left
        3: (0, 0, quadrant_width, quadrant_height),             # Top-left
        4: (quadrant_width, 0, width, quadrant_height)          # Top-right
    }
    return quadrants

def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        "orange": ((10, 100, 20), (25, 255, 255)),
        "green": ((36, 25, 25), (86, 255, 255)),
        "yellow": ((25, 50, 50), (35, 255, 255)),
        "white": ((0, 0, 200), (180, 20, 255)),
    }
    
    detected_balls = []
    
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detected_balls.append((color, (x + w // 2, y + h // 2)))
    return detected_balls

def get_quadrant(point, quadrants):
    x, y = point
    for quadrant, (x1, y1, x2, y2) in quadrants.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return quadrant
    return None

def track_events(video_path, output_video_path, output_text_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file '{video_path}' not found.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    quadrants = None
    
    event_log = []
    ball_positions = {}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Error opening video writer with path '{output_video_path}'.")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if quadrants is None:
            quadrants = define_quadrants(frame)
        
        detected_balls = detect_balls(frame)
        
        for color, position in detected_balls:
            quadrant = get_quadrant(position, quadrants)
            
            if color not in ball_positions:
                ball_positions[color] = quadrant
                event_log.append((frame_count / fps, quadrant, color, "Entry"))
                cv2.putText(frame, f"{color} Entry at Q{quadrant}", (position[0], position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif ball_positions[color] != quadrant:
                event_log.append((frame_count / fps, quadrant, color, "Entry"))
                event_log.append((frame_count / fps, ball_positions[color], color, "Exit"))
                ball_positions[color] = quadrant
                cv2.putText(frame, f"{color} Exit at Q{quadrant}", (position[0], position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for quadrant, (x1, y1, x2, y2) in quadrants.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for quadrant borders
            cv2.putText(frame, f"Q{quadrant}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    with open(output_text_path, 'w') as f:
        for log in event_log:
            f.write(f"{log[0]:.2f}, {log[1]}, {log[2]}, {log[3]}\n")

if __name__ == "__main__":
    video_path = "AI Assignment video.mp4"  # Path to the downloaded video
    output_video_path = r"D:\Projects\workspace\Ball Tracking & Recognition\output_video.mp4"  # Path to save the processed video
    output_text_path = r"D:\Projects\workspace\Ball Tracking & Recognition\output_text_file.txt"  # Path to save the event log

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    track_events(video_path, output_video_path, output_text_path)
