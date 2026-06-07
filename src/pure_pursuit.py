import numpy as np

def steering_angle(pos, yaw, look_pt, wheelbase, Ld):
    x = pos[0]
    y = pos[1]
    dx = look_pt[0] - x
    dy = look_pt[1] - y

    local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
    local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy
    alpha = np.arctan2(local_y, local_x)
    return np.arctan2(2 * wheelbase * np.sin(alpha), Ld)
