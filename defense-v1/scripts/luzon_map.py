# /// script
# requires-python = ">=3.12"
# dependencies = ["imagecodecs", "numpy", "tifffile"]
# ///
"""Build the map data for slide 3: Luzon's land, a typhoon track, and illustrative post dots.

Run with `uv run scripts/luzon_map.py` from the defense-v1 folder.
"""

import io
import json
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

COUNTRIES = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-10m.json"
POPULATION = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/PHL/phl_ppp_2020_1km_Aggregated_UNadj.tif"
OUTPUT = Path(__file__).parent.parent / "data" / "luzon-map.json"

WIDTH, HEIGHT = 540, 456
WEST, SOUTH, EAST, NORTH = 119.5, 12.45, 126.2, 18.75

SMALLEST_ISLAND = 2e-5

TRACK = [
    (127.8, 13.4), (127.4, 13.6), (127.0, 13.8), (126.6, 14.1), (126.2, 14.3), (125.9, 14.6), (125.6, 14.9), (125.2, 15.1), (124.8, 15.4),
    (124.3, 15.7), (123.9, 16.0), (123.5, 16.3), (123.2, 16.6), (122.9, 16.8), (122.6, 17.1),
    (122.2, 17.2), (121.7, 17.1), (121.2, 17.0), (120.7, 17.0), (120.3, 17.1), (120.0, 17.0),
    (119.8, 17.0),
]

PEOPLE_PER_DOT = 50_000
REQUEST_SHARE = 0.25


def fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def mercator(lon, lat):
    return math.radians(lon), math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


left, bottom = mercator(WEST, SOUTH)
right, top = mercator(EAST, NORTH)
scale = min(WIDTH / (right - left), HEIGHT / (top - bottom))
offset_x = (WIDTH - scale * (right - left)) / 2
offset_y = (HEIGHT - scale * (top - bottom)) / 2


def project(lon, lat):
    x, y = mercator(lon, lat)
    return round(offset_x + scale * (x - left), 1), round(offset_y + scale * (top - y), 1)


def decode_arcs(topology):
    (sx, sy), (tx, ty) = topology["transform"]["scale"], topology["transform"]["translate"]
    arcs = []
    for arc in topology["arcs"]:
        x = y = 0
        points = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            points.append((x * sx + tx, y * sy + ty))
        arcs.append(points)
    return arcs


def ring(arcs, indexes):
    points = []
    for index in indexes:
        arc = arcs[index] if index >= 0 else arcs[~index][::-1]
        points.extend(arc[1:] if points else arc)
    return points


def area(points):
    lat = math.radians(sum(p[1] for p in points) / len(points))
    twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    return abs(twice) / 2 * math.cos(lat) * math.radians(1) ** 2


def inside(point, points):
    x, y = point
    result = False
    for (xi, yi), (xj, yj) in zip(points, points[-1:] + points[:-1]):
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            result = not result
    return result


def path(points, closed):
    projected = [project(*p) for p in points]
    return "M" + "L".join(f"{x},{y}" for x, y in projected) + ("Z" if closed else "")


def smooth(points):
    p = [project(*point) for point in points]
    p = [p[0], *p, p[-1]]
    segments = [f"M{p[1][0]},{p[1][1]}"]
    for a, b, c, d in zip(p, p[1:], p[2:], p[3:]):
        c1 = (b[0] + (c[0] - a[0]) / 6, b[1] + (c[1] - a[1]) / 6)
        c2 = (c[0] - (d[0] - b[0]) / 6, c[1] - (d[1] - b[1]) / 6)
        segments.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {c[0]},{c[1]}")
    return "".join(segments)


topology = json.loads(fetch(COUNTRIES))
arcs = decode_arcs(topology)
philippines = next(g for g in topology["objects"]["countries"]["geometries"] if g["properties"]["name"] == "Philippines")
islands = [[ring(arcs, r) for r in polygon] for polygon in philippines["arcs"]]
islands = [polygon for polygon in islands if area(polygon[0]) > SMALLEST_ISLAND]


def on_land(point):
    return any(inside(point, outer) and not any(inside(point, hole) for hole in holes) for outer, *holes in islands)


with tifffile.TiffFile(io.BytesIO(fetch(POPULATION))) as tiff:
    page = tiff.pages[0]
    people = np.clip(page.asarray().astype(float), 0, None)
    cell = page.tags["ModelPixelScaleTag"].value[0]
    corner_lon, corner_lat = page.tags["ModelTiepointTag"].value[3:5]

rows, cols = np.nonzero(people)
lons = corner_lon + (cols + 0.5) * cell
lats = corner_lat - (rows + 0.5) * cell
shown = (lons > WEST - 0.5) & (lons < EAST + 0.5) & (lats > SOUTH - 0.7) & (lats < NORTH + 0.5)
lons, lats, weights = lons[shown], lats[shown], people[rows[shown], cols[shown]]
running = np.cumsum(weights)

generator = random.Random(7)
posts = []
while len(posts) < round(running[-1] / PEOPLE_PER_DOT):
    index = int(np.searchsorted(running, generator.random() * running[-1]))
    point = (lons[index] + (generator.random() - 0.5) * cell, lats[index] + (generator.random() - 0.5) * cell)
    if on_land(point):
        posts.append([*project(*point), int(generator.random() < REQUEST_SHARE)])

OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps({
    "land": "".join(path(ring_, closed=True) for polygon in islands for ring_ in polygon),
    "track": smooth(TRACK),
    "posts": posts,
}))
print(f"Wrote {OUTPUT} with {len(posts)} posts")
