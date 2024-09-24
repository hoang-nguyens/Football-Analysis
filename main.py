import cv2
from player_with_ball import PlayerAssigner
from speed_estimator import SpeedEstimator
from team_assigner import TeamAssigner
from track import Tracker
from utils import bbox_handle, video_handle
import numpy as np

def main():
    frames = video_handle.read_video(r"D:\Project\Football-Analysis\input_video\video_01.mp4")
    
    tracker = Tracker(r"D:\Project\Football-Analysis\pretrained_model\best.pt") 

    tracks = tracker.get_track_objects(frames)  

    tracker.add_position_to_tracks(tracks)
    #print(tracks['players'])
    #print(type(tracks))
    tracks['ball'] = tracker.fill_ball_data(tracks['ball'])

    speed_estimator = SpeedEstimator()
    speed_estimator.speed_estimator(tracks)
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[2], 
                                    tracks['players'][2])
   
    for frame_num, track in enumerate(tracks['players']):
        for player_id, player in track.items():
            team_id = team_assigner.get_player_teams(frames[frame_num], player['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team_id]
    
    
    playerwithball = PlayerAssigner()
    for frame_num, track in enumerate(tracks['players']):
       player_has_ball = playerwithball.assigner(track, tracks['ball'][frame_num][1]['bbox'])

       if player_has_ball != -1:
           tracks['players'][frame_num][player_has_ball]['has_ball'] = True
    

    output_video_frame = tracker.draw(frames, tracks)

    #output_video_frame = speed_estimator.draw_speed(output_video_frame, tracks)

    video_handle.save_frames_to_video(output_video_frame, r"D:\Project\Football-Analysis\output_video\output_video.avi")

if __name__ == '__main__':
    main()







