import numpy as np
from vincenty import vincenty
import math


class GpsAnomalyDetector:
    ATTACK_CODE = 1
    ALL_GOOD_CODE = 0
    SUSPICIOUS_CODE = 9

    def __init__(self, base_threshold=0.01):
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

    def __anomaly_detector(self, diff_vgps_euc):
        max_diff = max(diff_vgps_euc)
        # avg_diff = np.mean(diff_vgps_euc) # for future use
        if (max_diff > self.base_threshold) and (self.__suspicious):
            return self.ATTACK_CODE
        elif max_diff > self.base_threshold:
            self.__suspicious = True
            return self.SUSPICIOUS_CODE
        else:
            return self.ALL_GOOD_CODE

    def send_info(self, points_gps, points_pos):
        gps_v_d = self.__get_vincenty_distances(points_gps)
        pos_e_d = self.__get_euclidean_distances(points_pos)
        diff_vgps_euc = abs(gps_v_d - pos_e_d)
        code = self.__anomaly_detector(diff_vgps_euc)
        return code
