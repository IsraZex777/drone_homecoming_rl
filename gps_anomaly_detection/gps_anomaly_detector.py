import numpy as np
from vincenty import vincenty
import math


class GpsAnomalyDetector:
    def __init__(self, base_threshold=10):
        self.base_threshold = base_threshold
        self.__suspicious = False

    def __get_vincenty_distances(self, points_gps):
        gps_vincenty_distances = []
        for index in range(len(points_gps) - 1):
            vincenty_distance = vincenty(points_gps[index], points_gps[index + 1]) * 1000
            gps_vincenty_distances.append(vincenty_distance)

        return np.array(gps_vincenty_distances)

    def __get_euclidean_distances(self, points_pos):
        positions_euclidean_distances = []
        for index in range(len(points_pos) - 1):
            pos_start_x1, pos_start_y1 = points_pos[index]
            pos_end_x2, pos_end_y2 = points_pos[index + 1]
            euclidean_distance = math.sqrt(
                math.pow(pos_end_x2 - pos_start_x1, 2) + math.pow(pos_end_y2 - pos_start_y1, 2))
            positions_euclidean_distances.append(euclidean_distance)

        return np.array(positions_euclidean_distances)

    def is_under_attack(self, points_gps, points_pos):
        gps_v_d = self.__get_vincenty_distances(points_gps)
        pos_e_d = self.__get_euclidean_distances(points_pos)
        return abs(pos_e_d[0] - gps_v_d[0]) >= self.base_threshold
