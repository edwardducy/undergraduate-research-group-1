#import "figure-style.typ": *
#show: figure-style

#let dashed = (thickness: 0.8pt, dash: "dashed")
#let train-style = (inset: 8pt, stroke: dashed, corner-radius: 5pt)

#diagram(
  spacing: (0.3in, 0.28in),
  edge-stroke: 1pt,
  node(enclose: ((0, 0), (1, 0)), box(width: 3.5in, align(left)[
    *Example Taglish post* \
    "Kailangan namin ng tubig at pagkain dito sa Brgy. San Isidro, Rodriguez. Wala pang relief na dumarating."
  ]), ..box-style, name: <post>),
  node(enclose: ((0, 1), (1, 1)), box(width: 3.5in)[*Shared pretrained transformer encoder* \ Reads the post once for both tasks], ..box-style, name: <enc>),
  node((0, 2), [*Classification head*], width: 1.7in, ..box-style),
  node((1, 2), [*NER head*], width: 1.7in, ..box-style),
  node((0, 3), box(height: 0.85in, align(left + top)[
    *Humanitarian category* \
    Requests or urgent needs
  ]), width: 1.7in, ..box-style),
  node((1, 3), box(height: 0.85in, align(left + top)[
    *Named entities* \
    Brgy. San Isidro (location) \
    Rodriguez (location) \
    kailangan namin ng tubig at pagkain (request)
  ]), width: 1.7in, ..box-style),
  node(enclose: ((0, 4), (1, 4)), box(width: 3.5in)[*Multi-task optimizer* \ Balances the two tasks], ..train-style, name: <opt>),
  edge(<post>, <enc>, "-|>"),
  edge((0, 1), (0, 2), "-|>"),
  edge((1, 1), (1, 2), "-|>"),
  edge((0, 2), (0, 3), "-|>"),
  edge((1, 2), (1, 3), "-|>"),
  edge((0, 3), (0, 4), "-|>", stroke: dashed),
  edge((1, 3), (1, 4), "-|>", stroke: dashed),
  edge(<opt>, (2.15, 4), (2.15, 1), <enc>, "-|>", stroke: dashed),
  node((2.15, 2.5), align(center)[Updates \ during \ training], inset: 3pt, fill: white),
)
