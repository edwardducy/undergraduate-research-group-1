# /// script
# requires-python = ">=3.12"
# ///
"""Build the map for the dataset slide: the land around Luzon and the track of each cyclone.

Run with `uv run scripts/cyclone_map.py` from the defense-v1 folder.
It rewrites `land`, `frame` and each storm's `track` in data/cyclones.json and keeps every other field.

The tracks are the real best tracks from IBTrACS v04r01 (Knapp et al., 2010), read from NOAA's
Western Pacific file. The file is about 115 MB, so a run takes a little while.
"""

import csv
import io
import json
import math
import urllib.request
from pathlib import Path

COUNTRIES = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-10m.json"
IBTRACS = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
OUTPUT = Path(__file__).parent.parent / "data" / "cyclones.json"

# The visible window, in slide units, and the latitudes at its top and bottom.
# The window runs past the slide margins to the right and bottom edges of the slide.
FRAME = {"left": 420, "top": 56, "right": 962, "bottom": 500}
NORTH, SOUTH = 19.9, 9.4
# The longitude that sits at the right edge of the window.
EAST = 126.5

# Small enough to keep the Babuyan Islands north of Luzon, which Nando crossed.
SMALLEST_ISLAND = 1e-6

# The international name and season of each local storm name in IBTrACS.
NAMES = {
    "Rolly": ("GONI", 2020),
    "Ulysses": ("VAMCO", 2020),
    "Odette": ("RAI", 2021),
    "Karding": ("NORU", 2022),
    "Paeng": ("NALGAE", 2022),
    "Egay": ("DOKSURI", 2023),
    "Kristine": ("TRAMI", 2024),
    "Pepito": ("MAN-YI", 2024),
    "Nando": ("RAGASA", 2025),
    "Tino": ("KALMAEGI", 2025),
    "Uwan": ("FUNG-WONG", 2025),
}
# The 2026 storm has not happened yet, so its path is a placeholder, as (longitude, latitude).
PLACEHOLDER = {"2026 storm": [(130.0, 11.6), (126.0, 12.3), (124.3, 12.6), (122.6, 12.7), (121.0, 12.6), (119.3, 12.8), (113.0, 13.6)]}
# The part of each track that is kept, from where it nears the map until it first leaves this box.
TRACK_BOX = {"west": 112.0, "east": 130.0, "south": 8.0, "north": 21.5}


def fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def mercator(lon, lat):
    return math.radians(lon), math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


_, top = mercator(0, NORTH)
_, bottom = mercator(0, SOUTH)
scale = (FRAME["bottom"] - FRAME["top"]) / (top - bottom)
east, _ = mercator(EAST, 0)


def project(lon, lat):
    x, y = mercator(lon, lat)
    return round(FRAME["right"] + scale * (x - east), 1), round(FRAME["top"] + scale * (top - y), 1)


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


def visible(points, margin=20):
    xs, ys = zip(*(project(*p) for p in points))
    return (max(xs) > FRAME["left"] - margin and min(xs) < FRAME["right"] + margin
            and max(ys) > FRAME["top"] - margin and min(ys) < FRAME["bottom"] + margin)


def outline(points):
    projected = [project(*p) for p in points]
    # Drop points closer than half a unit to the last kept one, so the coast stays light.
    kept = [projected[0]]
    for point in projected[1:]:
        if math.dist(point, kept[-1]) >= 0.5:
            kept.append(point)
    return "M" + "L".join(f"{x},{y}" for x, y in kept) + "Z"


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
islands = [polygon for polygon in islands if area(polygon[0]) > SMALLEST_ISLAND and visible(polygon[0])]

data = json.loads(OUTPUT.read_text())
data["frame"] = FRAME
data["land"] = "".join(outline(r) for polygon in islands for r in polygon)


def best_tracks():
    """Read the six-hourly positions of each named storm from IBTrACS."""
    wanted = {key: name for name, key in NAMES.items()}
    found = {name: [] for name in NAMES}
    with urllib.request.urlopen(IBTRACS) as response:
        rows = csv.reader(io.TextIOWrapper(response, encoding="utf-8"))
        header = next(rows)
        next(rows)  # The second row holds the units.
        column = {name: index for index, name in enumerate(header)}
        for row in rows:
            key = (row[column["NAME"]], int(row[column["SEASON"]]))
            if key in wanted and row[column["ISO_TIME"]][11:13] in ("00", "06", "12", "18"):
                found[wanted[key]].append((float(row[column["LON"]]), float(row[column["LAT"]])))
    return found


def trim(points):
    """Keep the stretch of a track from where it enters the box until it first leaves it."""
    box = TRACK_BOX
    inside_box = [box["west"] < lon < box["east"] and box["south"] < lat < box["north"] for lon, lat in points]
    start = inside_box.index(True)
    end = inside_box.index(False, start) if False in inside_box[start:] else len(points)
    return points[max(start - 1, 0):end + 1]


TRACKS = {name: trim(points) for name, points in best_tracks().items()} | PLACEHOLDER
for storm in data["storms"]:
    storm["track"] = smooth(TRACKS[storm["name"]])
OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote {OUTPUT} with {len(islands)} islands")


# Check that every path crosses land inside the window, away from its faded edges.
def inside(point, points):
    x, y = point
    result = False
    for (xi, yi), (xj, yj) in zip(points, points[-1:] + points[:-1]):
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            result = not result
    return result


for name, points in TRACKS.items():
    crossed = 0
    for (lon_a, lat_a), (lon_b, lat_b) in zip(points, points[1:]):
        for step in range(200):
            point = (lon_a + (lon_b - lon_a) * step / 200, lat_a + (lat_b - lat_a) * step / 200)
            x, y = project(*point)
            shown = FRAME["left"] + 60 < x < FRAME["right"] - 30 and FRAME["top"] + 20 < y < FRAME["bottom"] - 20
            if shown and any(inside(point, polygon[0]) for polygon in islands):
                crossed += 1
    print(f"{name}: {crossed} samples over land in the window")
