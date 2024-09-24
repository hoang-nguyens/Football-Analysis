from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TeamAssigner():
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}
        
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters = 2, init = "k-means++", n_init = 5)

        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        image = image[0: image.shape[0] //2, :] # only take top half image

        kmeans = self.get_clustering_model(image)
        labels = kmeans.labels_

        clustered_image = labels.reshape(image.shape[0], image.shape[1])
        corner = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1,0], clustered_image[-1,-1]]
        
        non_player_cluster = max(set(corner), key = corner.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def assign_team_color(self, frame, players_track): # players_track is tracks['player'][frame_num]
        player_colors = []

        for track_id, track in players_track.items():
            bbox = track['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        kmeans = KMeans(n_clusters = 2, init = "k-means++", n_init = 5)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]
    
    def get_player_teams(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1

        self.player_team_dict[player_id] = team_id

        return team_id







