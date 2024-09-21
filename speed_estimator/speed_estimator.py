import cv2
from utils import get_foot, euclid_measure

class SpeedEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def speed_estimator(self, tracks):
        players = tracks['players']
        num_of_frames = len(players)

        for frame_num in range(0, num_of_frames, self.frame_window):
            last_frame = min(frame_num + self.frame_window, num_of_frames - 1)
            for track_id, _ in players[frame_num].items():
                if track_id not in players[last_frame].items():
                    continue

                start_position = tracks[frame_num][track_id]['position']
                end_position = tracks[frame_num][track_id]['position']

                if start_position is None or end_position is None:
                    continue

                distance = euclid_measure(start_position, end_position)
                time = (last_frame - frame_num) / self.frame_rate
                meters_per_second = distance / time
                kilometer_per_hour = meters_per_second * 3.6

                for frame in range(frame_num, last_frame):
                    tracks['player'][frame][track_id]['speed'] = kilometer_per_hour
    def draw_speed(self, frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees':
                    continue
                for track_id, track in object_tracks[frame_num].item():
                    if 'speed' in track:
                        speed = track['speed']
                    if speed is None: continue
                    bbox = tracks['bbox']
                    position = list(get_foot(bbox))
                    position[1] + 40
                    position = tuple(map(int, position))

                    cv2.putText(frame, f'{speed:.2f}', position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            output_video_frames.append(frame)
        return output_video_frames



