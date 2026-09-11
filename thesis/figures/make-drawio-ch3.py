#!/usr/bin/env python3
"""Generate editable draw.io sources for the chapter 3 figures.

One .drawio file per figure, into this folder. Open them in diagrams.net
(draw.io) to edit, then export PDF over the matching fig-*.pdf file so
the chapter render picks the new art up. The pymupdf script
(make-figures.py) regenerates interim renditions with the same names.

Rerun after editing: uv run python3 mise run figures
"""

from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent

STYLE_BOX = ("rounded=0;whiteSpace=wrap;html=1;fillColor=#F4F4F4;"
             "strokeColor=#000000;fontFamily=Helvetica;fontSize=11;fontStyle=1")
STYLE_EDGE = ("edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;"
              "strokeColor=#000000;fontFamily=Helvetica;fontSize=10")
STYLE_ELLIPSE = ("ellipse;whiteSpace=wrap;html=1;fillColor=#F4F4F4;"
                 "strokeColor=#000000;fontFamily=Helvetica;fontSize=11")
STYLE_ACTOR = ("shape=umlActor;verticalLabelPosition=bottom;"
               "verticalAlign=top;html=1;outlineConnect=0;"
               "strokeColor=#000000;fontFamily=Helvetica;fontSize=11")
STYLE_BOUNDARY = ("rounded=0;whiteSpace=wrap;html=1;verticalAlign=top;"
                  "fontStyle=1;fillColor=none;strokeColor=#000000;"
                  "fontFamily=Helvetica;fontSize=12")


class Diagram:
    def __init__(self, name, w, h):
        self.name = name
        self.w, self.h = w, h
        self.cells = []
        self.n = 1

    def node(self, value, x, y, w, h, style=STYLE_BOX):
        self.n += 1
        self.cells.append(
            f'<mxCell id="n{self.n}" value="{escape(value)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry" /></mxCell>')
        return f"n{self.n}"

    def edge(self, source, target, label=""):
        self.n += 1
        label_attr = f'value="{escape(label)}" ' if label else ""
        self.cells.append(
            f'<mxCell id="e{self.n}" {label_attr}style="{STYLE_EDGE}" '
            f'edge="1" parent="1" source="{source}" target="{target}">'
            f'<mxGeometry relative="1" as="geometry" /></mxCell>')

    def save(self):
        body = "".join(self.cells)
        xml = (
            '<mxfile host="app.diagrams.net">'
            f'<diagram id="{self.name}" name="{self.name}">'
            f'<mxGraphModel dx="1000" dy="700" grid="0" gridSize="10" '
            f'page="1" pageWidth="{self.w}" pageHeight="{self.h}">'
            f'<root><mxCell id="0" /><mxCell id="1" parent="0" />{body}'
            '</root></mxGraphModel></diagram></mxfile>')
        (HERE / f"{self.name}.drawio").write_text(xml, encoding="utf-8")


def pipeline():
    d = Diagram("fig-3-1-pipeline", 1160, 340)
    specs = [
        ("Baseline Analysis", "Corpus assembly&#10;STL baselines&#10;EW fine-tuning and&#10;encoder selection"),
        ("Design and Artifact Creation", "Hard-sharing trainer&#10;MTO configurations&#10;Gradient hooks&#10;Prototype service"),
        ("Empirical Benchmarking", "120 runs&#10;MLflow records&#10;Gradient diagnostics&#10;Efficiency timings"),
        ("Comparative Validation", "Bootstrap intervals&#10;Friedman-Nemenyi&#10;Profile analysis&#10;Transfer correlation"),
    ]
    ids = []
    for i, (title, body) in enumerate(specs):
        ids.append(d.node(f"<b>{title}</b><br>{body}", 20 + i * 290, 60, 250, 200))
    for a, b in zip(ids, ids[1:]):
        d.edge(a, b)
    d.save()


def model():
    d = Diagram("fig-3-2-model", 1160, 470)
    a = d.node("<b>Tweet text</b>", 20, 180, 160, 110)
    b = d.node("<b>Tokenizer</b><br>subword pieces", 230, 180, 170, 110)
    c = d.node("<b>Shared multilingual encoder</b><br>shared parameters", 450, 80, 200, 300)
    h1 = d.node("<b>Token head</b><br>BIO labels for LOC and INFRA", 750, 30, 340, 100)
    h2 = d.node("<b>Intent head</b><br>merged triage classes", 750, 180, 340, 100)
    h3 = d.node("<b>Urgency head</b><br>four-level scale, binary conversion", 750, 330, 340, 100)
    d.edge(a, b)
    d.edge(b, c)
    for h in (h1, h2, h3):
        d.edge(c, h)
    d.save()


def dataflow():
    d = Diagram("fig-3-3-dataflow", 1160, 420)
    s1 = d.node("<b>Replayed</b><br>event streams", 20, 60, 200, 100)
    s2 = d.node("<b>Manual entry</b>", 20, 220, 200, 100)
    svc = d.node("<b>FastAPI service</b><br>single-pass inference:&#10;entities, intent, urgency", 270, 100, 240, 180)
    gaz = d.node("<b>PSGC gazetteer</b><br>unmapped-span fallback", 560, 120, 200, 140)
    db = d.node("<b>PostgreSQL</b><br>incident store", 810, 120, 160, 140)
    dsh = d.node("<b>Dashboard</b><br>New, Acknowledged,&#10;Responding, Resolved", 990, 80, 160, 220)
    d.edge(s1, svc)
    d.edge(s2, svc)
    d.edge(svc, gaz)
    d.edge(gaz, db)
    d.edge(db, dsh)
    d.save()


def usecases():
    d = Diagram("fig-3-4-usecases", 1160, 640)
    actor = d.node("Municipal command operator", 40, 260, 60, 120, style=STYLE_ACTOR)
    d.node("Prototype triage service", 300, 20, 840, 600, style=STYLE_BOUNDARY)
    cases = ["Ingest replayed messages", "Monitor incident dashboard",
             "Acknowledge incident", "Mark incident responding",
             "Mark incident resolved", "View mapped locations"]
    for i, label in enumerate(cases):
        c = d.node(label, 340, 50 + i * 95, 760, 60, style=STYLE_ELLIPSE)
        d.edge(actor, c)
    d.save()


if __name__ == "__main__":
    pipeline()
    model()
    dataflow()
    usecases()
    print("draw.io sources written to", HERE)
