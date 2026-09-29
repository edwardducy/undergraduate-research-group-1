# Prose Style Rules

Derived by direct observation of five reference documents (MiMo engineering blog,
MOPD paper, Claude Sonnet 5.5 System Card, OpenAI GPT-6 Sol and Luna announcement,
OpenAI Navier-Stokes result post). Each rule states the required behavior for
manuscript prose. Manuscript mechanics (margins, font, spacing, black text,
APA 7th Edition) are governed by OUTLINE.md.

1. Open with context or a concrete event before stating the contribution or finding (Mimo: "Following the release of MiMo-V2.6, tool-call repetition emerged").
2. Make the study, not its writers, the actor of every sentence. Use "this study," "the analysis," "the model," or "the encoder" as the subject, and never use first-person pronouns (the system card's "We assess that..." becomes "This study assesses that...").
3. Put the paragraph's claim in its opening sentence and spend the rest of the paragraph supporting it.
4. Front-load purpose and transition clauses before the main clause ("To answer these questions, this study constructs...").
5. Chain sentences by opening follow-ups with "This" or "These" plus a summarizing noun ("This metric has two advantages").
6. Narrate experiments and events in past tense. State definitions, mechanisms, methods, and standing conclusions in present tense.
7. Use plain active verbs, and use the passive voice only when the actor is irrelevant. Never write agentless passives such as "results will be reported." Write "this study will report" instead.
8. Keep sentences mostly between 10 and 35 words, with a median near 20 and deliberate variation. Never write runs of three or more sentences under 15 words, which read as fragmented. Let the longest sentences carry parallel enumerations rather than digressions.
9. Write measured values as digits with decimals and units directly in the sentence ("weighted F1 between 0.768 and 0.781").
10. Attach a scale comparison to any headline number whose magnitude matters, and place it at the end of its sentence, after the claim it quantifies ("approximately $90,000, only 4% of the estimated cost").
11. Mark estimated quantities with about, roughly, or approximately, and keep measured values exact.
12. Report a change as a before-and-after or A-versus-B pair in one sentence ("from 13.45% to 3.83%").
13. State regressions and counter-evidence explicitly and quantified, in the same breath as the gains, pivoting with but, though, or however.
14. Grade comparisons with anchored relative terms ("within 1.1 percentage points of"), not bare adjectives ("better").
15. Frame interpretive conclusions as assessments ("the analysis suggests that"), and leave measured facts unqualified.
16. Define each technical term at first use, spell out abbreviations in parentheses, and distinguish neighboring concepts before analyzing them.
17. Use a colon only to introduce an inline list whose count the sentence just announced, to introduce a formally stated question (at most once per section), or in structural labels such as table stubs and figure captions. When a list is displayed as numbered or bulleted items, end the introducing sentence with a period instead. Never join two independent clauses with a colon. Use at most one colon per sentence.
18. Start bulleted items with a bold run-in label naming the claim, then explain it ("No exposure bias. The student is trained on its own rollouts").
19. Use parentheses for citations, counts, and short qualifying asides. Never use em dashes or nested commas for asides. If an aside needs a second clause of its own, promote it to a separate sentence.
20. Always use the serial comma in lists.
21. Use no exclamation marks or emoji, and use a question mark only for a question the text goes on to answer.
22. Avoid contractions.
23. Move dense multi-way comparisons into tables, and repeat only headline numbers in prose.
24. Keep paragraphs to roughly one to four sentences under descriptive noun-phrase or gerund headings.
25. Close a document or chapter with a forward-looking statement, not a recap.
26. Write headings in Title Case, matching the established chapter conventions ("Background of the Study").
27. Do not open sentences with And, But, or So. Pivot with However or Thus instead.
28. Announce each section's scope near its opening, as in "This section describes..." or "This chapter reviews...".
29. Call out every table or figure in prose with a present-tense reporting verb ("Table 3 presents...").
30. Keep citations and cross-references parenthetical, at the end of the clause they support.
31. Introduce every example with an explicit marker, "For example,", "such as", or "(e.g., ...)", never by bare juxtaposition.
32. Do not use semicolons in prose. Split the sentence in two, or move the aside into parentheses.
