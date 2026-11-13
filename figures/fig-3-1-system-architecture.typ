#import "figure-style.typ": *
#show: figure-style

#let part-width = 1.75in
#let part-height = 0.72in

#let part(pos, name, detail) = node(
  pos,
  [#strong(name) \ #text(size: 8.5pt, detail)],
  width: part-width,
  height: part-height,
  ..box-style,
)

#let store(pos, name, detail) = node(
  pos,
  [#strong(name) \ #text(size: 8.5pt, detail)],
  width: part-width,
  height: part-height + 0.04in,
  shape: fletcher.shapes.cylinder,
  inset: 8pt,
  stroke: 0.9pt,
)

#let system(pos, name) = node(pos, emph(name), width: part-width, height: 0.2in)

#let flow(..args, body) = edge(..args, "-|>", text(size: 8.5pt, body))

#diagram(
  spacing: (0.5in, 0.55in),
  edge-stroke: 0.9pt,

  node((0, 1), [#strong[X API] \ #text(size: 8.5pt)[External service]], width: 1in, ..box-style),

  system((1, 0.45), [Benchmarking System]),
  part((1, 1), [Post Collection], [Requests the posts from the X API]),
  part((2, 1), [Post Cleaning], [Filters the posts and replaces personal details]),
  part((3, 1), [Post Annotation], [doccano and GLiNER]),
  store((3, 2), [Dataset], [Training, validation, and test posts]),
  part((2, 2), [Model Training and Evaluation], [PyTorch, Transformers, and LibMTL]),
  store((1, 2), [Trained Models], [With their F1-scores, training time, and peak memory]),
  node(enclose: ((1, 0.45), (1, 1), (3, 1), (3, 2), (1, 2)), ..group-style, inset: 10pt, snap: false),

  system((1, 3.5), [Prototype Web Application]),
  part((3, 4), [Post Stream], [Replays the test posts one by one]),
  part((2, 4), [Server], [FastAPI, with the final model]),
  part((1, 4), [Web Interface], [React, with a map and a table]),
  store((1, 5.35), [Database], [PostgreSQL]),
  store((3, 5.35), [Place List], [Philippine places with coordinates]),
  node(enclose: ((1, 3.5), (1, 4), (3, 4), (1, 5.35), (3, 5.35)), ..group-style, inset: 10pt, snap: false),

  flow((0, 1), (1, 1))[],
  flow((1, 1), (2, 1))[],
  flow((2, 1), (3, 1))[],
  flow((3, 1), (3, 2))[],
  flow((3, 2), (2, 2))[],
  flow((2, 2), (1, 2))[],
  flow((1, 2), (1, 2.9), (2, 2.9), (2, 4), label-pos: 0.78, label-side: left)[Final Model],
  flow((3, 2), (3, 4), label-side: left, label-pos: 0.3)[Test Posts],
  flow((3, 4), (2, 4))[],
  edge((2, 4), (1, 4), "<|-|>"),
  edge((2, 4), (2, 4.62), (1, 4.62), (1, 5.35), "<|-|>"),
  edge((2, 4), (2, 4.62), (3, 4.62), (3, 5.35), "<|-|"),
)
