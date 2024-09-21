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
                bbox = detect[0].to_list()
                cls_id = detect[3]
                track_id = detect[4]

                if cls_id == class_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}
                if cls_id == class_name_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for detect in detection_supervison:
                bbox = detect[0].to_list()
                cls_id = detect[3]

                if cls_id == class_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}
        if track_paths is not None:
            with open(track_paths, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def fill_ball_data(self, ball_position): # handle with null data of ball position
        # ball_positon = tracks['ball']
        position = [x.get(1, {}).get('bbox', {}) for x in ball_position]

        df_position = pd.DataFrame(position, columns=['x1', 'y1', 'x2', 'y2'])

        df_position = df_position.interpolate()
        df_position = df_position.bfill()

        ball_position = [ {1:{'bbox':x}} for x in df_position.to_numpy().to_list()]

        return ball_position
    
    def add_position_to_tracks(self, tracks): # beside tracks[name][n_frame][track_id] we have dict with 1 element is 'bbox':bbox, now add 'position'
        for object, object_tracks in tracks.items():
            for frame_num, tracks in enumerate(object_tracks):
                for track_id, track_bbox in tracks.items():
                    bbox = track_bbox['bbox']
                    if object == 'ball':
                        tracks[object][frame_num][track_id]['position'] = get_center(bbox)
                    if object == 'player':
                        tracks[object][frame_num][track_id]['position'] = get_foot(bbox)
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y_ellipse = bbox[3]
        x_ellipse, _ = get_center(bbox)
        major_axis = get_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_ellipse, y_ellipse),
            axes = (major_axis, major_axis * 0.6),
            angle = 0, #  flat ellipse / no rotation
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType= cv2.LINE_4
        )
        if track_id is not None:
            x1_rectangle = (x_ellipse + 10)
            x2_rectangle = (x_ellipse - 10)
            y1_rectangle = (y_ellipse + 25)
            y2_rectangle = (y_ellipse - 25)

            cv2.rectangle(
                frame,
                pt1 = (x1_rectangle, y1_rectangle),
                pt2 = (x2_rectangle, y2_rectangle),
                color = color,
                thickness = -1
            )

            cv2.putText(
                frame,
                f'track_id',
                (x1_rectangle + 12, y1_rectangle + 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = bbox[1]
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
    
    def draw(self, video_frame, tracks):
        output_video_frame = []
        for frame_num , frame in enumerate(output_video_frame):
            frame = frame.copy()

            for track_id, player in tracks['players'][frame_num].items():
                color = player.get('team_color', (0,0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], color)
            
            for track_id, referee in tracks['referees'][frame_num].items():
                color = (0, 255, 255)
                frame = self.draw_ellipse(frame, referee['bbox'], color)
            for track_id, ball in tracks['ball'][frame_num].items():
                color = (0, 255, 0)
                frame = self.draw_triangle(frame,ball['bbox'], color)
            output_video_frame.append(frame)
        return output_video_frame









