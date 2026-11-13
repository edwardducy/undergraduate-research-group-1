#import "@preview/cetz:0.3.4": canvas, draw
#set page(width: auto, height: auto, margin: 4pt)
#set text(font: "Times New Roman", size: 8pt, hyphenate: false)
#set par(justify: false, leading: 0.4em)

#let small(body) = text(size: 7pt, body)

#canvas(length: 1in, {
  import draw: *
  let sw = 0.8pt
  set-style(stroke: sw)

  let W = 2.7
  let G = 0.11
  let PT = 0.11
  let PB = 0.24
  let FG = 0.1
  let FX = 2.8
  let SX = 2.12
  let SW = 1.15
  let arrow(..pts) = line(..pts.pos(), mark: (end: ">", fill: black, scale: 0.6), stroke: sw)
  let both(a, b) = line(a, b, mark: (start: ">", end: ">", fill: black, scale: 0.6), stroke: sw)

  let step(x, y, w, h, body) = {
    rect((x - w / 2, y), (x + w / 2, y - h))
    content((x, y - h / 2), box(width: (w - 0.1) * 1in, align(center, body)))
  }
  let cyl(x, y, w, h, body) = {
    let e = 0.07
    line((x - w / 2, y - e), (x - w / 2, y - h + e))
    line((x + w / 2, y - e), (x + w / 2, y - h + e))
    arc((x + w / 2, y - h + e), start: 0deg, stop: -180deg, radius: (w / 2, e))
    circle((x, y - e), radius: (w / 2, e))
    content((x, y - e - h / 2 + 0.01), box(width: (w - 0.1) * 1in, align(center, body)))
  }
  let io(x, y, w, h, body) = {
    let s = 0.1
    line((x - w / 2 + s, y), (x + w / 2, y), (x + w / 2 - s, y - h), (x - w / 2, y - h), close: true)
    content((x, y - h / 2), box(width: (w - 0.3) * 1in, align(center, body)))
  }
  let frame(top, bottom, label) = {
    rect((-FX, top), (FX, bottom), stroke: 0.6pt)
    content((-FX + 0.1, bottom + 0.06), anchor: "south-west", text(size: 7.5pt, style: "italic", label))
  }

  let y = 0
  io(0, y, 2.6, 0.36)[*Posts matching the search keywords* \ #small[(X API)]]
  let y-in-bottom = y - 0.36

  let f1-top = y-in-bottom - 0.15
  y = f1-top - PT
  let coll = y
  step(0, y, W, 0.33)[Collection of posts in proportion to each day's number of matching posts]
  y -= 0.33 + G
  let filt = y
  step(0, y, W, 0.33)[Filtering by the inclusion and exclusion criteria, and replacement of personal details with placeholders]
  y -= 0.33 + G
  let ann = y
  step(0, y, W, 0.44)[Annotation of categories by two annotators and an adjudicator, and of entity spans by GLiNER \ #small[(Cohen's kappa per batch)]]
  y -= 0.44 + G
  let split = y
  step(0, y, W, 0.22)[Split into training, validation, and test posts]
  y -= 0.22 + G
  let ds = y
  cyl(0, y, 1.2, 0.34)[Labeled dataset]
  y -= 0.34
  let f1-bottom = y - PB
  frame(f1-top, f1-bottom, [Data preparation])

  arrow((0, y-in-bottom), (0, coll))
  arrow((0, coll - 0.33), (0, filt))
  arrow((0, filt - 0.33), (0, ann))
  arrow((0, ann - 0.44), (0, split))
  arrow((0, split - 0.22), (0, ds))

  let f2-top = f1-bottom - FG
  let ds-bottom = y
  y = f2-top - PT
  let tune = y
  step(0, y, W, 0.44)[Fine-tuning of the pretrained encoder \ #small[(SEA-LION-ModernBERT-300M)] \ #small[(learning rate tuning, then main runs)]]
  arrow((0, ds-bottom), (0, tune))
  y -= 0.44 + G
  let models = y
  step(-0.7, y, 1.3, 0.33)[Single-task models \ #small[(one per task)]]
  step(0.7, y, 1.3, 0.33)[Candidate models \ #small[(one per optimizer)]]
  arrow((-0.7, tune - 0.44), (-0.7, models))
  arrow((0.7, tune - 0.44), (0.7, models))
  y -= 0.33 + G
  let saved = y
  let sv-w = 2.0
  let sv-h = 0.42
  cyl(0, y, sv-w, sv-h)[Saved models and logs \ #small[(F1-scores, training time, and peak memory)]]
  arrow((-0.7, models - 0.33), (-0.45, saved - 0.02))
  arrow((0.7, models - 0.33), (0.45, saved - 0.02))
  y -= sv-h + G
  let sel = y
  step(0, y, W, 0.33)[Optimizer selection on validation posts \ #small[(highest mean $Delta_"MTL"$)]]
  arrow((0, saved - sv-h), (0, sel))
  let lc-h = 0.44
  let lc-top = sel + (lc-h - 0.33) / 2
  step(-SX, lc-top, SW, lc-h)[Learning curve of the single-task models \ #small[(validation posts)]]
  arrow((-sv-w / 2, saved - sv-h / 2), (-SX, saved - sv-h / 2), (-SX, lc-top))
  y -= 0.33 + G
  let fm = y
  cyl(0, y, 1.2, 0.34)[Final model]
  arrow((0, sel - 0.33), (0, fm))
  let ev-x = 1.75
  let ev-w = 1.75
  step(ev-x, fm - 0.005, ev-w, 0.33)[Final evaluation on test posts \ #small[(paired t-tests and error analysis)]]
  arrow((0.6, fm - 0.17), (ev-x - ev-w / 2, fm - 0.17))
  y -= 0.34
  let fm-bottom = y
  let f2-bottom = y - PB
  frame(f2-top, f2-bottom, [Model training and selection])

  let f3-top = f2-bottom - FG
  let ret = f3-top - PT
  let rh = 0.33
  step(0, ret, W, rh)[Retrieval of new posts and labeling by the final model \ #small[(started by the researcher)]]
  arrow((0, fm-bottom), (0, ret))
  let in-x = -SX
  let in-w = SW
  io(in-x, ret - rh / 2 + 0.16, in-w, 0.32)[New posts \ #small[(X API)]]
  arrow((in-x + in-w / 2 - 0.05, ret - rh / 2), (-W / 2, ret - rh / 2))

  let row = ret - rh - 0.22
  let db-x = -1.85
  let pw = 1.5
  let ch = 0.36
  let bh = 0.22
  let bt = row - (ch - bh) / 2
  cyl(db-x, row, pw, ch)[PostgreSQL database]
  step(0, bt, pw, bh)[API server (FastAPI)]
  step(-db-x, bt, pw, bh)[Web interface (React)]
  let mid = bt - bh / 2
  both((db-x + pw / 2, mid), (-pw / 2, mid))
  both((pw / 2, mid), (-db-x - pw / 2, mid))
  let elbow = (ret - rh + row) / 2
  arrow((0, ret - rh), (0, elbow), (db-x, elbow), (db-x, row))
  let f3-bottom = row - ch - PB
  frame(f3-top, f3-bottom, [Prototype web application])

  let ob = f3-bottom - 0.08
  let out = f3-bottom - 0.24
  arrow((-db-x, bt - bh), (-db-x, ob), (0, ob), (0, out))
  io(0, out, 2.6, 0.46)[*Category and entities of each post,* \ *filtered by category or entity type,* \ *for responders*]
})
