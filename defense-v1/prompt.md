Build Isaiah's parts of the mock defense in defense-v1/slides.md.

CONTEXT (current deck)
- Page 1: title slide (layout: title). Keep as is.
- Page 2: <ModelDiagram /> (clicks: 13). Kyle's title explanation. Keep as is.
- Page 3: <RiskMap /> (clicks: 5). Already covers the FIRST paragraph of
  Isaiah's Background section (1st in disaster risk, 8–9 cyclones/year,
  too many posts, responders need a sorting tool). Keep as is.

SOURCE OF TRUTH
defense-v1/script.md, only the "### Isaiah" blocks under these sections.
Slide numbers are the 11-slide outline in
canvas_docs/05-module-4/02-m-mock-defense-guidelines.md, not deck page numbers.
- Slide 2, Background and Problem Statement → Isaiah's 2nd paragraph
  (one tool must give report type + details; most PH papers do one task;
  Urdaneta City system uses an LLM; smaller trained model is the other
  option; no Taglish dataset labeled for both). Add one slide right after
  <RiskMap />.
- Slide 5, Expected Scientific Contribution → Isaiah's block (three
  beneficiaries: LGUs/disaster offices, local universities, typhoon-hit
  communities; deliverables: dataset + guidelines, final model + benchmark
  results, code + prototype, X posts shared as IDs).
- Slide 8, Development and Testing Schedule → Isaiah's block (feasibility:
  training pilot fits laptop memory, fixed workload of 92 tuning + 51 main
  runs, X API spending capped at 200 USD with a pilot search).
Yngwie's halves of slides 2, 5 and 8 are out of scope. Put Isaiah's slides
where the outline order puts them, and add an HTML comment on each new slide
naming its outline slide number and section, e.g. <!-- Slide 5 · Isaiah -->.

CONTENT RULES
- Match the script's tone and claims. Add no claim, number, or source the
  script does not state.
- Short bullets or labels, not paragraphs (rubric rewards "bite-sized" text).
  The 6x6 rule and other lengths in canvas_docs are NOT requirements (AGENTS.md).
- Body text: no I/we/our/us/you/your. No semicolons, colons, or em dashes.
  CEFR B2 English for a reader outside computer science.

BUILD RULES
- Use only existing pieces: layouts/title.vue (only where its left/right
  grid fits), style.css tokens, ModelDiagram.vue, RiskMap.vue, and
  composables/useClickTimeline.ts. Write no new animation code.
- If a reveal is needed, use Slidev's built-in v-click / v-clicks.
- Set clicks: in each new slide's frontmatter to the exact number of reveal
  steps (0 or omitted if none).
- Do not edit script.md, the components, the composable, or the title slide.

WHEN DONE
List each slide added with its outline slide number and the script section
and paragraph it came from, and flag any line where the script was too long
to fit and had to be cut.
