import numpy as np 

def cumulative_arc_length(pts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0,0], np.cumsum(seg)])

def track_heading(points, index):
    j = (index + 1) % len(points)
    d = points[j] - points[index]
    return np.arctan2(d[1], d[0])

def cross_track_error(points, pos, index):
    th = track_heading(points, index)
    dx = pos[0] - points[index][0]
    dy = pos[1] - points[index][1]
    return -np.sin(th) * dx + np.cos(th) * dy  # signed lateral offset from the line

# make the cubic bezier curve
def cubic_bezier(P0, P1, P2, P3, n):
    t = np.linspace(0, 1, n, endpoint=False)[:, None]  # (n,1) so it broadcasts over x,y
    return ((1-t)**3 * P0 + 3*(1-t)**2 * t * P1
            + 3*(1-t) * t**2 * P2 + t**3 * P3) 

def make_track(waypoints, samples_per_seg=200, tension=0.0):
    wp = np.asarray(waypoints, dtype=float)
    n = len(wp) # num of curve segs
    s = (1 - tension) / 6 # tension scale factor
    segments = []
    for i in range(n):
        p0 = wp[i]
        p3 = wp[(i+1) % n] # modulo for last points so p0 = p3
        prev = wp[(i-1) % n]
        next = wp[(i+2) % n]

        # Catmull-Rom rule
        p1 = p0 + s * (p3 - prev)
        p2 = p3 - s * (next - p0)
        
        segments.append(cubic_bezier(p0, p1, p2, p3, samples_per_seg))
    return np.concatenate(segments, axis=0)

def nearest_index(pts, pos):
    return int(np.argmin(np.sum((pts - np.asarray(pos))**2, axis=1)))

def lookahead_point(pts, pos, Ld, start_idx):
    i = start_idx
    dist = 0.0
    n = len(pts)
    while dist < Ld:
        j = (i + 1) % n
        dist += np.linalg.norm(pts[j] - pts[i])
        i = j
    return pts[i], i

# waypoints = [(0, 0), (30, 5), (40, 25), (20, 40), (-10, 30), (-20, 5)]
# track = make_track(waypoints)
# kappa = curvature(track)
