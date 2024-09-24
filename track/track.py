from ultralytics import YOLO
import supervision as sv
import pickle
import pandas as pd
from utils import get_center, get_foot, get_width
import cv2
import numpy as np

class Tracker:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)
        self.track = sv.ByteTrack()

    def detect_frame(self, frames):
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detect = self.yolo.predict(frames[i: i + batch_size])
            detections += detect
        return detections

    def get_track_objects(self, frames, track_paths = None):
        detections = self.detect_frame(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            detection_supervison = sv.Detections.from_ultralytics(detection)
            detection_tracks = self.track.update_with_detections(detection_supervison)

            class_name = detection.names # return dict that keys are number, and values is classname
            class_name_inv = {v:k for k, v in class_name.items()}

            for object_ind, object_id in enumerate(detection_supervison.class_id):
                if class_name[object_id] == 'goalkeeper':
                    detection_supervison.class_id[object_ind] = class_name_inv['player']
            
            for detect in detection_tracks:
                bbox = detect[0].tolist()
                cls_id = detect[3]
                track_id = detect[4]

                if cls_id == class_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}
                if cls_id == class_name_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for detect in detection_supervison:
                bbox = detect[0].tolist()
                cls_id = detect[3]

                if cls_id == class_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}
        if track_paths is not None:
            with open(track_paths, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def fill_ball_data(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def add_position_to_tracks(self, tracks): # beside tracks[name][n_frame][track_id] we have dict with 1 element is 'bbox':bbox, now add 'position'
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_bbox in track.items():
                    bbox = track_bbox['bbox']
                    if object == 'ball':
                        position = get_center(bbox)
                        
                    else:
                        position = get_foot(bbox)
                    tracks[object][frame_num][track_id]['position']= position
       
    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y_ellipse = int(bbox[3])
        x_ellipse, _ = get_center(bbox)
        x_ellipse = int(x_ellipse)
        major_axis = get_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_ellipse, y_ellipse),
            axes = (int(major_axis), int(major_axis * 0.35)),
            angle = 0, #  flat ellipse / no rotation
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType= cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_ellipse - rectangle_width//2
        x2_rect = x_ellipse + rectangle_width//2
        y1_rect = (y_ellipse - rectangle_height//2) +15
        y2_rect = (y_ellipse + rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          -1)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center(bbox)
        triangle_point = np.array([
            [x, y],
            [x + 10, y - 15],
            [x - 10, y - 15]
        ])

        cv2.drawContours(
            frame, [triangle_point], 0, color, -1
        )
        cv2.drawContours(
            frame, [triangle_point], 0, (0, 0, 0), 2
        )
        return frame
    
    def draw(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames









