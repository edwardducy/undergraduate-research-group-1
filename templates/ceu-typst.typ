#import "@preview/versatile-apa:7.2.0": versatile-apa
#let to-string(content) = {
  if content.has("text") { content.text }
  else if content.has("children") { content.children.map(to-string).join("") }
  else if content.has("body") { to-string(content.body) }
  else if content == [ ] { " " }
  else { str(content) }
}

#show: versatile-apa.with(
  font-size: 12pt,
)

// CEU binding override: 1.5" left, 1.0" others (APA default is 1" all)
#set page(
  paper: "us-letter",
  margin: (left: 1.5in, top: 1in, right: 1in, bottom: 1in),
)

// CEU typography: Times New Roman 12pt with Liberation fallback (metric-identical)
#set text(
  font: ("Times New Roman", "Liberation Serif", "Libertinus Serif", "DejaVu Serif", "New Computer Modern"),
  size: 12pt,
  lang: "en",
  region: "US",
)

// Draft readability: visible blank line between paragraphs (APA has only indent) + justified
#set par(justify: true, spacing: 2.0em)
#set text(hyphenate: auto)

// CEU overrides: Chapter 1 16pt, headers 14pt, subheaders not italicized, tables/figures 10pt, inline placement
#show heading.where(level: 1): set text(size: 16pt, weight: "bold")
#show heading.where(level: 2): it => {
  if to-string(it.body).contains("References") {
    // References heading itself stays 14pt, following pars get hanging indent
    set text(size: 14pt, weight: "bold")
    it
    show par: set par(first-line-indent: 0pt, hanging-indent: 0.5in, leading: 1.5em, spacing: 1.5em, justify: false)
    show par: set text(size: 12pt)
  } else {
    set text(size: 14pt, weight: "bold")
    it
  }
}
#show heading.where(level: 3): set text(size: 12pt, style: "normal", weight: "bold")
#show heading.where(level: 4): it => {
  set text(size: 12pt, style: "normal", weight: "bold")
  block(it.body)
}
#show heading.where(level: 5): it => {
  set text(size: 12pt, style: "normal", weight: "bold")
  block(it.body)
}
#show table: set text(size: 10pt)
#show table.header: set text(weight: "bold")
#show figure.caption: set text(size: 10pt)
#set figure(placement: none)
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: image): set block(breakable: true)

$body$
