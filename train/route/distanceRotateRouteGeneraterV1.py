import numpy as np
import random
from mylib.config import Configurable
from mylib.config import save

class DistanceRotateRouteGeneraterV1(Configurable):
    x_range: save
    y_range: save
    path_length: save
    dist_min: save
    dist_max: save
    angle_min: save
    angle_max: save

    def __init__(self, x_range, y_range, path_length, dist_min=0, dist_max=5, angle_min=0, angle_max=360):
        self.x_range = int(x_range)
        self.y_range = int(y_range)
        self.path_length = path_length
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.angle_min = angle_min
        self.angle_max = angle_max

        self.__pre_angle = 0


    def get_next_position(self, pre_x, pre_y, recursion=0):
        r = random.randint(self.dist_min, self.dist_max)
        angle_dif = random.uniform(self.angle_min/2, self.angle_max/2) * random.choice([-1, 1])
        angle = self.__pre_angle + angle_dif

        if angle >= 360:
            angle -= 360
        elif angle < 0:
            angle += 360

        x = pre_x + r * np.cos(np.radians(angle))
        y = pre_y + r * np.sin(np.radians(angle))

        # x, yをx_range, y_range内の最も近い点(整数)に修正
        x = round(min(self.x_range - 1, max(0, x)))
        y = round(min(self.y_range - 1, max(0, y)))

        # 同じ座標になったら再計算(上限10回)
        if x == pre_x and y == pre_y:
            if recursion > 10:
                return None, None
            x, y = self.get_next_position(pre_x, pre_y, recursion + 1)
        
        if x != None:
            # 修正後の点で角度を再計算
            self.__pre_angle = np.degrees(np.arctan2(y - pre_y, x - pre_x))

        return x, y
    

    def get_route(self):
        while True:
            # Generate random starting point
            start_x = random.randint(0, self.x_range - 1)
            start_y = random.randint(0, self.y_range - 1)

            self.__pre_angle = random.randint(0, 360)
    
            # Initialize route with starting point
            route = [[start_x], [start_y]]
        
        
            for _ in range(self.path_length - 1):
                next_x, next_y = self.get_next_position(route[0][-1], route[1][-1])
                if next_x == None:
                    break

                route[0].append(next_x)
                route[1].append(next_y)
                    
            else:
                return route
    
    def get_route_list(self, num_routes):
        routes = [self.get_route() for _ in range(num_routes)]    
        # print(routes[0]) 
        return routes
    




