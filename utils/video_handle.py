import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()

        if not ret: break
        frames.append(frame)
    print(len(frames))
    return frames

import cv2

def save_frames_to_video(output_frames, video_path, fps=24, frame_size=(640, 480)):
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in output_frames:
        resized_frame = cv2.resize(frame, frame_size)
        out.write(resized_frame)

    out.release()

