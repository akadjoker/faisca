"""A ClipGT folder for a made-up two-lane rural road, for hdmap_ref.py to draw.

Same tables and columns as lucent tools/omnidreams/osi_to_clipgt.py writes (egomotion_estimate,
obstacle, lane_line, road_boundary, the calibration copied from a real scene). FLU world, the
ego rig (rear axle centre on the ground) starting at the origin heading +x.

The road: a centre line of slowly varying curvature, lanes --lane metres wide, a dashed white
centre line, solid white edge lines, road boundaries --shoulder metres outside them; the ego in
the right lane at --speed km/h, sampled at 10 Hz; oncoming cars in the left lane, the first after
--first-car seconds, then every --car-every seconds.
Usage: synthetic_road.py <out dir> --calibration calibration_estimate.parquet [options]
"""
import argparse
import math
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--calibration", required=True)
parser.add_argument("--seconds", type=float, default=64.0)
parser.add_argument("--speed", type=float, default=70.0, help="km/h")
parser.add_argument("--lane", type=float, default=3.5)
parser.add_argument("--shoulder", type=float, default=0.8)
parser.add_argument("--offset", type=float, default=0.0, help="ego lateral offset in its lane, + left")
parser.add_argument("--first-car", type=float, default=6.0)
parser.add_argument("--car-every", type=float, default=7.0)
parser.add_argument("--car-speed", type=float, default=80.0, help="km/h")
parser.add_argument("--straight", type=float, default=150.0, help="metres before the first bend")
args = parser.parse_args()

speed = args.speed / 3.6
length = speed * args.seconds + 400.0
step = 0.5
count = int(length / step) + 1
s = np.arange(count) * step
# curvature: straight first, then gentle bends (radius down to ~500 m)
curvature = np.where(s < args.straight, 0.0,
                     0.002 * np.sin(2 * math.pi * (s - args.straight) / 700.0)
                     + 0.0008 * np.sin(2 * math.pi * (s - args.straight) / 260.0 + 1.0))
heading = np.concatenate([[0.0], np.cumsum(curvature[:-1] * step)])
x = np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * step)]) - 20.0
y = np.concatenate([[0.0], np.cumsum(np.sin(heading[:-1]) * step)])
centre = np.stack([x, y], axis=1)
left_normal = np.stack([-np.sin(heading), np.cos(heading)], axis=1)


def offset_line(d):
    return centre + left_normal * d


def at(distance, lateral):
    i = min(max(distance / step, 0.0), count - 1.001)
    k = int(i)
    t = i - k
    p = (1 - t) * offset_line(lateral)[k] + t * offset_line(lateral)[k + 1]
    h = (1 - t) * heading[k] + t * heading[k + 1]
    return p, h


def quat_yaw(h):
    return {"x": 0.0, "y": 0.0, "z": math.sin(h / 2), "w": math.cos(h / 2)}


clip = "synthetic_rural_road"
clipgt = os.path.join(args.out, "clipgt")
os.makedirs(clipgt, exist_ok=True)


def write(name, keys, values):
    pq.write_table(pa.table({"key": pa.array(keys), name: pa.array(values)}), os.path.join(clipgt, f"{name}.parquet"))


def rail(points):
    return [{"x": float(p[0]), "y": float(p[1]), "z": 0.0} for p in points]


start_s = 20.0  # the ego starts 20 m along the road
ego_lateral = -args.lane / 2 + args.offset
times = np.arange(0.0, args.seconds + 1e-6, 0.1)
ego_keys, ego_values = [], []
for t in times:
    p, h = at(start_s + speed * t, ego_lateral)
    ego_keys.append({"clip_id": clip, "timestamp_micros": int(round(t * 1e6)), "label_class_id": "osi:host_vehicle"})
    ego_values.append({"name": "host_vehicle", "location": {"x": float(p[0]), "y": float(p[1]), "z": 0.0},
                       "orientation": quat_yaw(h)})
write("egomotion_estimate", ego_keys, ego_values)

car_speed = args.car_speed / 3.6
obstacle_keys, obstacle_values = [], []
track = 0
first = args.first_car
while first < args.seconds:
    # an oncoming car that passes the ego `first` seconds in: where is it at time t
    meet = start_s + speed * first
    for t in times:
        distance = meet + car_speed * (first - t)
        if distance < start_s + speed * t - 40.0 or distance > start_s + speed * t + 250.0:
            continue
        p, h = at(distance, args.lane / 2)
        obstacle_keys.append({"clip_id": clip, "timestamp_micros": int(round(t * 1e6)),
                              "label_class_id": "osi:moving_object", "label_id": f"car{track}:{int(round(t * 1e6))}"})
        obstacle_values.append({"trackline_id": f"car{track}", "center": {"x": float(p[0]), "y": float(p[1]), "z": 0.75},
                                "size": {"x": 4.6, "y": 1.85, "z": 1.5}, "orientation": quat_yaw(h + math.pi),
                                "category": "automobile"})
    track += 1
    first += args.car_every
if obstacle_keys:
    write("obstacle", obstacle_keys, obstacle_values)

key = lambda name: {"clip_id": clip, "label_class_id": "osi:lane_boundary", "map_id": name, "map_id_version": ""}
pieces = range(0, count - 1, 400)  # 200 m pieces (ludus uses the first 256 vertices of a polyline)
lines, edges = [], []
for i in pieces:
    j = min(i + 401, count)
    lines.append((key(f"centre{i}"), {"line_rail": rail(offset_line(0.0)[i:j]), "styles": ["LONG_DASHED_SINGLE"],
                                      "colors": ["WHITE"]}))
    for side, sign in (("left", 1.0), ("right", -1.0)):
        lines.append((key(f"{side}{i}"), {"line_rail": rail(offset_line(sign * args.lane)[i:j]),
                                          "styles": ["SOLID_SINGLE"], "colors": ["WHITE"]}))
        edges.append((key(f"{side}_edge{i}"), {"category": "road_boundary",
                                               "location": rail(offset_line(sign * (args.lane + args.shoulder))[i:j])}))
# ludus resamples lines every 0.2 m and keeps 256 vertices: 400 points of 0.5 m would be cut; split again
def split(items, field):
    out = []
    for k, v in items:
        r = v[field]
        for a in range(0, len(r) - 1, 48):  # 24 m: a whole number of 12 m dash periods
            piece = dict(v)
            piece[field] = r[a:a + 49]
            kk = dict(k)
            kk["map_id"] = f"{k['map_id']}_{a}"
            out.append((kk, piece))
    return out


lines = split(lines, "line_rail")
edges = split(edges, "location")
write("lane_line", [k for k, _ in lines], [v for _, v in lines])
write("road_boundary", [k for k, _ in edges], [v for _, v in edges])
shutil.copyfile(args.calibration, os.path.join(clipgt, "calibration_estimate.parquet"))
print(f"wrote {clipgt}: {len(times)} ego poses over {args.seconds} s at {args.speed} km/h, {track} oncoming cars, "
      f"{len(lines)} lane line pieces, {len(edges)} road boundary pieces")
