from utils import get_center, euclid_measure

class PlayerAssigner():
    def __init__(self):
        self.max_distance = 70
    
    def assigner(self, players, ball_bbox):# player here are track['player'][frame_num]
        ball = get_center(ball_bbox)
        min_distance = 9999
        player_with_ball = -1

        for track_id, track in players.items():
            player_bbox = track['bbox']

            left = euclid_measure((player_bbox[0], player_bbox[3]), ball)
            right = euclid_measure((player_bbox[2], player_bbox[3]), ball)
            distance = min(left, right)

            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    player_with_ball = track_id
        return player_with_ball


