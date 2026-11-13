#import "figure-style.typ": *
#show: figure-style

#let stick-figure = box(width: 0.3in, height: 0.42in, {
  let s = 0.9pt
  place(dx: 0.1in, dy: 0in, circle(radius: 0.05in, stroke: s))
  place(line(start: (0.15in, 0.1in), end: (0.15in, 0.26in), stroke: s))
  place(line(start: (0.04in, 0.16in), end: (0.26in, 0.16in), stroke: s))
  place(line(start: (0.15in, 0.26in), end: (0.06in, 0.42in), stroke: s))
  place(line(start: (0.15in, 0.26in), end: (0.24in, 0.42in), stroke: s))
})

#let actor(pos, name, label-dx: 0, label-dy: 0.8) = {
  let (x, y) = pos
  (node(pos, stick-figure, inset: 0pt), node((x + label-dx, y + label-dy), strong(name), height: 0pt))
}

#let hand(pos, side) = (rel: (side * 0.11in, -0.05in), to: pos)

#let use-case-width = 1.9in

#let left-tip(pos) = (rel: (-use-case-width / 2, 0pt), to: pos)
#let right-tip(pos) = (rel: (use-case-width / 2, 0pt), to: pos)

#let use-case(pos, body) = node(
  pos,
  body,
  shape: fletcher.shapes.ellipse,
  width: use-case-width,
  height: 0.32in,
  stroke: 0.8pt,
  fill: luma(96%),
)

// Each use case with the actors linked to it, in the order of the use case table.
#let cases = (
  [View posts on a map],
  [View posts in a table],
  [View the details of a post],
  [Filter posts by category],
  [Filter posts by entity type],
  [Filter posts by location],
  [Search posts by keyword],
  [View a summary of posts],
)

#let responder-pos = (0, 3.5)

#diagram(
  spacing: (1.1in, 0.1in),
  edge-stroke: 0.8pt,

  node((1, -1), strong[Prototype Web Application]),
  ..cases.enumerate().map(((y, c)) => use-case((1, y), c)),
  node(enclose: ((1, -1), (1, cases.len() - 1)), inset: 10pt, stroke: 1pt, fill: luma(98%), snap: false),

  ..actor(responder-pos, label-dx: -0.45, label-dy: 0)[Responder],

  ..range(0, cases.len()).map(y => edge(hand(responder-pos, 1), left-tip((1, y)), snap-to: none)),
)
