# Slides Planning

Working plan for the title defense deck. The deck itself lives in slides/title-defense-slides.md. This file records what each slide shows and how the presenter delivers it.

Timing per part:

- Part 1: 3 to 4 minutes, 9 slides.
- Part 2: 3 to 4 minutes, 4 slides.
- Part 3: 5 to 7 minutes, 8 slides.
- Part 4: 3 to 4 minutes, 8 slides.
- Part 5: 2 to 3 minutes, 6 slides.
- Total: 16 to 22 minutes, 35 slides.

## Part 1, Slide 1: Title

- Shows: the study title, the event name, the group name.
- Present: read the title once, slowly. State the part and its time budget.

## Part 1, Slide 2: The Three Outputs

- Shows: a three-row table of the outputs, then the joint triage sentence.
- Present: name each output together with its formal task name. Connect the outputs to the word "triage" in the title.

## Part 1, Slide 3: Why Disasters

- Shows: the two motivation sentences.
- Present: emphasize the four actions that people report. Use this slide to lead into Taglish.

## Part 1, Slide 4: Why Taglish

- Shows: the three Taglish sentences.
- Present: stress the mixing within one word. Then introduce the example tweet.

## Part 1, Slide 5: The Example Tweet

- Shows: the tweet, the three readings, the "baha" note.
- Present: read the tweet aloud in the original. Give the three readings slowly. Pause at the word "ma-evacuate" before you continue.

## Part 1, Slide 6: The Multilingual Transformer Encoder

- Shows: one sentence.
- Present: slow down here. The audience must understand what the encoder does before the next slide.

## Part 1, Slide 7: One Shared Encoder Instead of Three

- Shows: two side-by-side diagrams, three encoders against one shared encoder, with a closing line on local deployment.
- Present: contrast the two columns aloud. End on the local deployment point.

## Part 1, Slide 8: The Conflict

- Shows: the pull bullets and the NER-loses consequence.
- Present: emphasize that which task loses is an open question. Deliver the rescue-delay consequence deliberately.

## Part 1, Slide 9: The Research Question

- Shows: the definition line and the question.
- Present: read the definition, pause, then read the question as the question of the whole study. Hand off to Part 2.

## Part 2, Slide 10: What Exists

- Shows: three literature bullets.
- Present: one sentence per bullet, no details. Name the studies, without citation codes.

## Part 2, Slide 11: What Is Missing

- Shows: the three gaps and the CALCS conditional result.
- Present: emphasize the phrases "no published study" and "no dataset" and let the CALCS bullet support the word "conditional" at the end.

## Part 2, Slide 12: The Two Options and Their Costs

- Shows: a two-row table of the options and their costs, then the guess-or-copy line and the open question.
- Present: slow down. This slide is the gap in one picture. Land on the sentence "which one is an open question" as the close.

## Part 2, Slide 13: Our Answer: A Controlled Benchmark

- Shows: two sentences.
- Present: deliver this as the transition to the methodology. Point forward to Part 3.

## Part 3, Slide 14: The System Pipeline

- Shows: the diagram.
- Present: walk the diagram top to bottom with a pointer. Repeat the line "one pass per tweet" as the key point.

## Part 3, Slide 15: The Example Tweet Through the Diagram

- Shows: the four walkthrough bullets.
- Present: connect "ma-evacuate" splitting at the hyphen. End on the single pass being cheaper.

## Part 3, Slide 16: Goal 1: Adapt to Fragmented Code-Switched Text

- Shows: the three goal bullets, without citation codes.
- Present: name the goal, the evidence, and that this goal needs no new evidence from our own results.

## Part 3, Slide 17: Goal 2: Balance the Three Outputs

- Shows: the four goal bullets, baselines included.
- Present: name the baselines explicitly. Say "the established ones and the newer ones" without method names.

## Part 3, Slide 18: How Balancing Works

- Shows: two mini-diagrams side by side, static equal pulls against dynamic adjusted pulls with one thicker arrow, and one caption under each.
- Present: introduce the pull metaphor here. The audience will hear it again in Part 4.

## Part 3, Slide 19: The Training Loop

- Shows: the six lines, with line 5 highlighted.
- Present: read all six lines. Point at line 5. Say that the protocol keeps every other line identical.

## Part 3, Slide 20: The Example Tweet Through the Loop

- Shows: the line-by-line bullets.
- Present: follow the tweet line by line. Land on the thousands-of-training-signals imbalance.

## Part 3, Slide 21: From Measurement to Method

- Shows: a five-step flow from the comparison to the benchmarked method, then the closing two lines.
- Present: stress the line "targeted adaptation, not a new general-purpose algorithm" as the scope of the work. Deliver the formula-measurement line as the close of the part.

## Part 4, Slide 22: The Data and the Event Splits

- Shows: the dataset line, a training-testing split diagram, and the split explanation.
- Present: state the undecided dataset scale honestly. Explain why event splits are the realistic choice.

## Part 4, Slide 23: The Controlled Conditions

- Shows: the four condition bullets.
- Present: emphasize the same conditions for every method and the significance testing.

## Part 4, Slide 24: The Per-Output Scores

- Shows: the per-answer scoring rule, two bars for the common and the rare answers, then the NER and reporting rules.
- Present: explain the averaging with the example. Stress that we report the three scores separately.

## Part 4, Slide 25: The Extra Checks

- Shows: the four check bullets.
- Present: mention that the switch points are the hardest part of the Taglish text.

## Part 4, Slide 26: The Cost Metrics

- Shows: a table of the measurements and their settings, then the practicality line.
- Present: name the two laptops. Connect the numbers to local deployment.

## Part 4, Slide 27: The Gradient Diagnostics

- Shows: the three diagnostic bullets.
- Present: reuse the pull metaphor. Cite GradNorm and PCGrad.

## Part 4, Slide 28: The Decision Rule

- Shows: the three rule bullets.
- Present: state the rule slowly. This rule is the link from the benchmark to the method.

## Part 4, Slide 29: The Success Criterion

- Shows: the three criterion bullets.
- Present: state what success means, and what we measure on the other two outputs.

## Part 5, Slide 30: Feasibility: The Resources Exist

- Shows: the five resource bullets.
- Present: reassure the panel on resources. Mention the annotation protocol and the source notes.

## Part 5, Slide 31: The Timeline: UR1

- Shows: a three-milestone timeline.
- Present: list them quickly, then point to UR2.

## Part 5, Slide 32: The Timeline: UR2

- Shows: a five-milestone timeline.
- Present: end on the public release.

## Part 5, Slide 33: Risks and Controls

- Shows: a three-row table pairing each risk with its control.
- Present: pair each risk with its control.

## Part 5, Slide 34: Four Contributions

- Shows: the numbered list.
- Present: read all four slowly. These are the deliverables.

## Part 5, Slide 35: Closing

- Shows: the two steps and the final line.
- Present: read the final line last, then stop.
