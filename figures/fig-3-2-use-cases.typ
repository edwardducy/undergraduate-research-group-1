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
  [Retrieve posts from X],
  [Open the prototype],
  [View the labeled posts],
  [View the category of a post],
  [View the entities of a post],
  [Filter posts by category],
  [Filter posts by entity type],
  [Combine filters],
  [Clear filters],
)

#let researcher-pos = (0, 0)
#let responder-pos = (0, 4.5)
#let x-api-pos = (2, 0)

#diagram(
  spacing: (1.1in, 0.1in),
  edge-stroke: 0.8pt,

  node((1, -1), strong[Prototype Web Application]),
  ..cases.enumerate().map(((y, c)) => use-case((1, y), c)),
  node(enclose: ((1, -1), (1, cases.len() - 1)), inset: 10pt, stroke: 1pt, fill: luma(98%), snap: false),

  ..actor(researcher-pos, label-dx: -0.45, label-dy: 0)[Researcher],
  ..actor(responder-pos, label-dx: -0.45, label-dy: 0)[Responder],
  ..actor(x-api-pos, label-dx: 0.4, label-dy: 0)[X API],

  edge(hand(researcher-pos, 1), left-tip((1, 0)), snap-to: none),
  ..range(1, cases.len()).map(y => edge(hand(responder-pos, 1), left-tip((1, y)), snap-to: none)),
  edge(hand(x-api-pos, -1), right-tip((1, 0)), snap-to: none),
)
