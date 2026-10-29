# /// script
# requires-python = ">=3.12"
# ///
"""Build the data for the prototype slide: illustrative posts in Marikina and the map tiles under them.

Run with `uv run scripts/marikina_posts.py` from the defense-v1 folder. The tiles are saved in the
deck, so the slide's map works without an internet connection.
"""

import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "marikina-posts.json"
MAP = ROOT / "public" / "map"

STYLE_URL = "https://tiles.openfreemap.org/styles/positron"
ZOOMS = range(11, 14)
GLYPH_RANGES = ("0-255", "256-511", "8192-8447")
WEST, SOUTH, EAST, NORTH = 121.05, 14.595, 121.15, 14.695

BARANGAYS = {
    "Barangka": (14.63279, 121.08221, 1.5),
    "Calumpang": (14.62413, 121.09325, 1.5),
    "Concepcion Uno": (14.65105, 121.10412, 1.5),
    "Concepcion Dos": (14.64059, 121.12082, 0.7),
    "Fortune": (14.66329, 121.12736, 0.7),
    "Industrial Valley": (14.62710, 121.08000, 1.5),
    "Jesus de la Peña": (14.63609, 121.09211, 2),
    "Malanday": (14.65016, 121.09375, 3),
    "Marikina Heights": (14.65011, 121.11714, 0.7),
    "Nangka": (14.67291, 121.10921, 2.5),
    "Parang": (14.65886, 121.11173, 1),
    "San Roque": (14.62573, 121.10288, 1),
    "Santa Elena": (14.63278, 121.09749, 1),
    "Santo Niño": (14.64006, 121.09684, 2),
    "Tañong": (14.63455, 121.08583, 1.5),
    "Tumana": (14.66041, 121.10078, 3),
}

CATEGORIES = [
    ("Caution and advice", 0.12, 0.5),
    ("Displaced people and evacuations", 0.11, 0.9),
    ("Infrastructure and utility damage", 0.10, 0.9),
    ("Injured or dead people", 0.04, 0.85),
    ("Missing or found people", 0.03, 0.85),
    ("Not humanitarian", 0.05, 0.2),
    ("Other relevant information", 0.08, 0.6),
    ("Requests or urgent needs", 0.18, 0.9),
    ("Rescue, volunteering, or donation effort", 0.14, 0.85),
    ("Sympathy and support", 0.15, 0.3),
]
POSTS = 640

SPREAD = 0.0022


def tile(lon, lat, zoom):
    n = 2**zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


FEATURED = [
    ("Displaced people and evacuations", "Nangka", "Mahigit 200 pamilya na ang nasa evacuation center sa Nangka Elementary School"),
    ("Infrastructure and utility damage", "Tumana", "Bagsak ang tulay sa Brgy. Tumana, hindi madaanan ng sasakyan"),
    ("Rescue, volunteering, or donation effort", "Malanday", "Nag-deploy na ng rescue boats ang Marikina Rescue 161 sa Malanday"),
    ("Caution and advice", "Santo Niño", "Lumikas na po ang mga nakatira malapit sa ilog sa Santo Niño, tumataas pa ang tubig"),
    ("Requests or urgent needs", "Concepcion Dos", "Need ng tubig at pagkain dito sa Brgy. Concepcion Dos, Marikina. Stranded kami sa 2nd floor"),
]

generator = random.Random(7)
names = list(BARANGAYS)
weights = [BARANGAYS[name][2] for name in names]
stream = []
for _ in range(POSTS - len(FEATURED)):
    category = generator.choices(range(len(CATEGORIES)), [share for _, share, _ in CATEGORIES])[0]
    if generator.random() > CATEGORIES[category][2]:
        stream.append([None, None, category])
        continue
    lat, lon, _ = BARANGAYS[generator.choices(names, weights)[0]]
    stream.append([round(generator.gauss(lat, SPREAD), 5), round(generator.gauss(lon, SPREAD), 5), category])

categories = [name for name, _, _ in CATEGORIES]
OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps({
    "bounds": [[SOUTH, WEST], [NORTH, EAST]],
    "categories": categories,
    "stream": stream,
    "featured": [{"lat": BARANGAYS[place][0], "lon": BARANGAYS[place][1], "category": categories.index(category), "text": text} for category, place, text in FEATURED],
}, ensure_ascii=False))
located = sum(1 for post in stream if post[0] is not None) + len(FEATURED)
print(f"Wrote {OUTPUT} with {located} posts on the map and {POSTS - located} without a location")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "thesis-defense-slides"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def save(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(url))
        time.sleep(0.05)


style = json.loads(fetch(STYLE_URL))
tiles = json.loads(fetch(style["sources"]["openmaptiles"]["url"]))["tiles"][0]
fonts = {font for layer in style["layers"] for font in layer.get("layout", {}).get("text-font", [])}
sprite = style["sprite"]
style["sources"] = {"openmaptiles": {"type": "vector", "tiles": ["{base}map/tiles/{z}/{x}/{y}.pbf"], "minzoom": 0, "maxzoom": max(ZOOMS), "attribution": '<a href="https://openfreemap.org">OpenFreeMap</a> © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}}
style["layers"] = [layer for layer in style["layers"] if layer.get("source") in (None, "openmaptiles")]
style["glyphs"] = "{base}map/fonts/{fontstack}/{range}.pbf"
style["sprite"] = "{base}map/sprites/ofm"
MAP.mkdir(parents=True, exist_ok=True)
(MAP / "style.json").write_text(json.dumps(style))

for zoom in ZOOMS:
    left, top = tile(WEST, NORTH, zoom)
    right, bottom = tile(EAST, SOUTH, zoom)
    for x in range(left - 1, right + 2):
        for y in range(top - 1, bottom + 2):
            save(tiles.format(z=zoom, x=x, y=y), MAP / "tiles" / str(zoom) / str(x) / f"{y}.pbf")
for font in fonts:
    for glyphs in GLYPH_RANGES:
        save(f"https://tiles.openfreemap.org/fonts/{urllib.parse.quote(font)}/{glyphs}.pbf", MAP / "fonts" / font / f"{glyphs}.pbf")
for suffix in ("", "@2x"):
    for kind in ("json", "png"):
        save(f"{sprite}{suffix}.{kind}", MAP / "sprites" / f"ofm{suffix}.{kind}")
print(f"Saved the map in {MAP}")
