# CEU institutional style

Rules required by the institution on top of APA 7. These override or extend
`thesis/docs/apa7-mechanics.md` where the mechanics doc notes house exceptions:

- **Person** — the ban on first- and second-person pronouns stays even though
  APA 4.16 permits them for the writers' own actions; phrase own actions
  impersonally ("This chapter reviews…").
- **SentenceLength** — the 30-word cap; APA itself sets no cap (4.6).

Also institutional but not mechanically checkable (manual review): numbered
headings (2.1, 2.1.1 format retained despite APA 2.27) and impersonal active
voice for own actions.

Layered in `.vale.ini` as `BasedOnStyles = Vale, APA7, CEU`.
