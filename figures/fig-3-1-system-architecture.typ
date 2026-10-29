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
  let G = 0.15
  let PT = 0.11
  let PB = 0.26
  let FG = 0.1
  let FX = 2.8
  let arrow(a, b) = line(a, b, mark: (end: ">", fill: black, scale: 0.6), stroke: sw)
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
  io(0, y, 2.6, 0.36)[*Taglish typhoon posts* \ #small[(X API, with public Facebook pages as fallback)]]
  let y-in-bottom = y - 0.36

  let f1-top = y-in-bottom - 0.15
  y = f1-top - PT
  let clean = y
  step(0, y, W, 0.22)[Cleaning and Taglish identification]
  y -= 0.22 + G
  let ann = y
  step(0, y, W, 0.33)[Annotation by two annotators in doccano, with entity spans drafted by GLiNER]
  y -= 0.33 + G
  let split = y
  step(0, y, W, 0.22)[Split into training, validation, and test posts]
  y -= 0.22 + G
  let ds = y
  cyl(0, y, 1.2, 0.34)[Labeled dataset]
  y -= 0.34
  let f1-bottom = y - PB
  frame(f1-top, f1-bottom, [Data preparation])

  arrow((0, y-in-bottom), (0, clean))
  arrow((0, clean - 0.22), (0, ann))
  arrow((0, ann - 0.33), (0, split))
  arrow((0, split - 0.22), (0, ds))

  let f2-top = f1-bottom - FG
  let ds-bottom = y
  y = f2-top - PT
  let tune = y
  step(0, y, W, 0.33)[Fine-tuning of the pretrained encoder \ #small[(SEA-LION-ModernBERT-300M)]]
  arrow((0, ds-bottom), (0, tune))
  y -= 0.33 + G
  let models = y
  step(-0.7, y, 1.3, 0.33)[Single-task models \ #small[(one per task)]]
  step(0.7, y, 1.3, 0.33)[Candidate models \ #small[(one per optimizer)]]
  arrow((-0.7, tune - 0.33), (-0.7, models))
  arrow((0.7, tune - 0.33), (0.7, models))
  y -= 0.33 + G
  let saved = y
  cyl(0, y, 1.6, 0.34)[Saved models and logs]
  arrow((-0.7, models - 0.33), (-0.4, saved - 0.02))
  arrow((0.7, models - 0.33), (0.4, saved - 0.02))
  y -= 0.34 + G
  let sel = y
  step(0, y, W, 0.33)[Optimizer selection on validation posts \ #small[(ANOVA, Dunnett's test, training time)]]
  arrow((0, saved - 0.34), (0, sel))
  y -= 0.33 + G
  let ev = y
  step(0, y, W, 0.33)[Final evaluation on test posts \ #small[(paired t-tests, storage, processing time)]]
  arrow((0, sel - 0.33), (0, ev))
  y -= 0.33 + G
  let fm = y
  cyl(0, y, 1.2, 0.34)[Final model]
  arrow((0, ev - 0.33), (0, fm))
  y -= 0.34
  let f2-bottom = y - PB
  frame(f2-top, f2-bottom, [Model training and selection])

  let f3-top = f2-bottom - FG
  let fm-bottom = y
  y = f3-top - PT - 0.05
  let api = y
  step(0, y, W, 0.22)[API server (FastAPI) running the final model]
  arrow((0, fm-bottom), (0, api))
  cyl(W / 2 + 0.3 + 0.55, y + 0.05, 1.1, 0.32)[PostgreSQL database]
  both((W / 2, y - 0.11), (W / 2 + 0.3, y - 0.11))
  y -= 0.22 + 0.24
  let web = y
  step(0, y, W, 0.22)[Web interface (React)]
  both((0, api - 0.22), (0, web))
  y -= 0.22
  let f3-bottom = y - PB
  frame(f3-top, f3-bottom, [Prototype web application])

  let out = f3-bottom - 0.15
  arrow((0, y), (0, out))
  io(0, out, 2.6, 0.36)[*Posts sorted by category, with entities,* \ *for DRRMO staff*]
})
