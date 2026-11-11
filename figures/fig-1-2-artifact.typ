#import "figure-style.typ": *
#show: figure-style

#let dashed = (thickness: 0.8pt, dash: "dashed")

#diagram(
  spacing: (0.25in, 0.3in),
  edge-stroke: 1pt,
  node(enclose: ((0, 0), (2, 0)), box(width: 2.6in, align(left)[
    *Example Taglish post* \
    "Kailangan namin ng tubig at pagkain dito sa Brgy. San Isidro, Rodriguez. Wala pang relief na dumarating."
  ]), ..box-style, name: <post>),
  node(enclose: ((0, 1), (2, 1)), box(width: 2.6in)[*Shared pretrained transformer encoder* \ Reads the post once for both tasks], ..box-style, name: <enc>),
  node((0, 2), [*Classification head*], width: 1.75in, ..box-style),
  node((2, 2), [*NER head*], width: 1.75in, ..box-style),
  node((1, 2), [*Multi-task optimizer* \ #text(size: 8.5pt)[training only]], width: 1.0in, inset: 6pt, stroke: dashed, corner-radius: 5pt),
  node((0, 3), box(height: 0.9in, align(left + top)[
    *Humanitarian category* \
    Requests or urgent needs
  ]), width: 1.75in, ..box-style),
  node((2, 3), box(height: 0.9in, align(left + top)[
    *Named entities* \
    Brgy. San Isidro (location) \
    Rodriguez (location) \
    kailangan namin ng tubig at pagkain (request)
  ]), width: 1.75in, ..box-style),
  edge(<post>, <enc>, "-|>"),
  edge((0, 1), (0, 2), "-|>", [whole post], label-side: right),
  edge((2, 1), (2, 2), "-|>", [each token], label-side: left),
  edge((0, 2), (0, 3), "-|>"),
  edge((2, 2), (2, 3), "-|>"),
  edge((0, 2), (1, 2), "-|>", [loss], stroke: dashed, label-side: right),
  edge((2, 2), (1, 2), "-|>", [loss], stroke: dashed, label-side: left),
  edge((1, 2), (1, 1), "-|>", [updates], stroke: dashed, label-side: left),
)
