#let title-page(title: none, authors: ()) = page(header: none, {
  set align(center)
  set par(first-line-indent: 0in, justify: false)

  v(0.25in)
  strong(title)

  v(1fr)
  [
    An Undergraduate Research Proposal Presented to the \
    Department of Computer Science and Information Technology \
    Centro Escolar University Makati
  ]

  v(1fr)
  [
    In Partial Fulfillment of the Requirements for the Degree \
    Bachelor of Science in Computer Science
  ]

  v(1fr)
  [by]
  v(0.1in)
  authors.join(linebreak())

  v(1fr)
  [
    Engr. Ma Christina A. Florentino \
    Advisor
  ]

  v(1fr)
  [November 2026]

  counter(page).update(0)
})
