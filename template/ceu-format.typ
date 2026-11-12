#import "title-page.typ": title-page

#let single = 1.15em
#let double = 2 * single
#let line-gap = double - 1em

#let front-matter() = {
  set par(first-line-indent: 0in, justify: false)
  let title(t) = block(width: 100%, below: line-gap, align(center, strong(t)))
  let entry(indent, body, loc) = {
    let page-number = numbering(loc.page-numbering(), ..counter(page).at(loc))
    par(first-line-indent: (amount: indent, all: true), hanging-indent: indent)[
      #link(loc)[#body#box(width: 1fr, repeat[.])#page-number]
    ]
  }
  let trim(c) = if c.func() == [].func() and c.children.len() > 0 and c.children.last() == [ ] {
    c.children.slice(0, -1).join()
  } else { c }
  let chapter-number(loc) = {
    let numbered = query(heading.where(level: 1).before(loc)).filter(h => h.numbering != none)
    let pattern = if numbered.len() > 0 { numbered.last().numbering } else { "1" }
    numbering(pattern, counter(heading).at(loc).first())
  }
  let float-list(kind, supplement) = context {
    for f in query(figure.where(kind: kind)) {
      let loc = f.location()
      let n = chapter-number(loc) + "." + str(f.counter.at(loc).first())
      entry(0in, [#strong[#supplement #n.] #trim(f.caption.body)], loc)
    }
  }

  title[TABLE OF CONTENTS]
  context {
    let chapters-started = false
    for h in query(heading).filter(h => h.body != [] and h.level <= 3) {
      let loc = h.location()
      if h.level == 1 and h.numbering == none {
        entry(0in, upper(h.body), loc)
      } else if h.level == 1 {
        if not chapters-started {
          par[CHAPTER]
          chapters-started = true
        }
        entry(0.25in, [#chapter-number(loc). #h.body], loc)
      } else {
        let number = numbering(h.numbering, ..counter(heading).at(loc))
        entry(if h.level == 2 { 0.25in } else { 0.5in }, [#number #h.body], loc)
      }
    }
  }

  pagebreak()
  title[LIST OF TABLES]
  float-list("quarto-float-tbl", "Table")

  pagebreak()
  title[LIST OF FIGURES]
  float-list("quarto-float-fig", "Figure")
}

// Called by Quarto for the `appendices` of the book. Each appendix is lettered
// (Appendix A), and its sections and tables follow the letter (A.1, Table A.1).
#let appendices(title, doc, hide-parent: false) = {
  counter(heading).update(0)
  set heading(numbering: "A.1")
  show heading.where(level: 1): it => {
    if it.numbering == none {
      // The break comes first even when the heading is hidden, so that its
      // entry in the table of contents points to the page of Appendix A.
      pagebreak(weak: true)
      if not hide-parent { align(center, it.body) }
    } else {
      pagebreak(weak: true)
      counter(figure.where(kind: "quarto-float-fig")).update(0)
      counter(figure.where(kind: "quarto-float-tbl")).update(0)
      align(center)[
        Appendix #counter(heading).display("A") \
        #it.body
      ]
    }
  }
  set figure(numbering: n => numbering("A.1", counter(heading).get().first(), n))
  doc
}

#let ceu-format(title: none, authors: (), body) = {
  set document(title: title, author: authors)

  set page(
    paper: "us-letter",
    margin: (left: 1.5in, right: 1in, top: 1in, bottom: 1in),
    header: context align(right, counter(page).display()),
    footer: none,
    numbering: none,
  )

  set text(font: "Times New Roman", size: 12pt, lang: "en", region: "US")

  set text(top-edge: 1em, bottom-edge: 0em)
  set par(
    leading: line-gap,
    spacing: line-gap,
    justify: true,
    first-line-indent: (amount: 0.5in, all: true),
  )
  set list(spacing: line-gap)
  set enum(spacing: line-gap)

  set heading(numbering: "1.1")
  show heading: set text(size: 12pt, weight: "bold")
  show heading: set block(above: line-gap, below: line-gap)

  show heading.where(body: []): set heading(numbering: none)
  show heading.where(body: []): none

  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    counter(figure.where(kind: "quarto-float-fig")).update(0)
    counter(figure.where(kind: "quarto-float-tbl")).update(0)
    set align(center)
    if it.numbering != none [
      Chapter #counter(heading).display("1") \
      #it.body
    ] else {
      it.body
    }
  }

  set figure(numbering: n => numbering("1.1", counter(heading).get().first(), n))
  set figure(gap: line-gap)
  show figure: set block(above: line-gap, below: line-gap)
  show figure.where(kind: "quarto-float-tbl"): set block(breakable: true)
  show figure.caption: it => align(left, block(width: 100%, sticky: true)[
    #strong[#it.supplement #context it.counter.display(it.numbering)] \
    #it.body
  ])

  set table(stroke: 0.5pt, inset: (x: 0.5em, top: 0.45em, bottom: 0.6em))
  set table.hline(stroke: 0.5pt)
  set table.header(repeat: false)
  set table.cell(breakable: false)
  show table: set par(justify: false, first-line-indent: 0in, leading: single - 1em, spacing: single - 1em)
  show table: set text(hyphenate: true)

  show table: it => {
    if it.at("label", default: none) == <ceu-rebuilt> { return it }
    let ncols = if type(it.columns) == array { it.columns.len() } else { it.columns }
    let is-head = c => c.func() == table.header or c.func() == table.hline
    let head = it.children.filter(is-head).map(c => {
      if c.func() == table.header { table.header(..c.children.map(x => strong(x))) } else { c }
    })
    let cells = it.children.filter(c => not is-head(c))
    let args = (columns: it.columns, align: it.align)
    let keep = 2 * ncols
    if cells.len() <= keep + ncols {
      return block(breakable: false, [#table(..args, ..head, ..cells)<ceu-rebuilt>])
    }
    set block(spacing: 0pt)
    block(breakable: false, [#table(..args, ..head, ..cells.slice(0, keep))<ceu-rebuilt>])
    [#table(..args, stroke: (x, y) => (
      left: 0.5pt, right: 0.5pt, bottom: 0.5pt, top: if y == 0 { none } else { 0.5pt },
    ), ..cells.slice(keep))<ceu-rebuilt>]
  }

  show math.equation.where(block: true): set block(above: line-gap, below: line-gap)

  show <refs>: set par(first-line-indent: 0in, hanging-indent: 0.5in)

  title-page(title: title, authors: authors)

  {
    set page(numbering: "i")
    front-matter()
  }
  set page(numbering: "1")
  counter(page).update(1)
  body
}
