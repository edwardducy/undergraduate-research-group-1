---
theme: none
colorSchema: light
fonts:
  sans: Inter
  provider: none
title: Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets
transition: view-transition
defaults:
  transition: slide-left
layout: title
---

# Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets

::left::

### Research Group

- Isaiah Raphael Almuena
- Prinz Edward Ducyogen
- Carlo Miguel Geronimo
- Kyle Uri Gabriel Lu
- Yngwie Sebastian Patricio

::right::

### Research Advisor

Engr. Ma Christina A. Florentino

### Date of Defense

October 27, 2026

---
clicks: 13
---

<ModelDiagram />

---
clicks: 5
---

<RiskMap />

---
clicks: 15
---

<div class="stack">
  <ApproachDiagram />
  <NegativeTransfer :first="6" />
</div>

---

<div class="quote">

### Statement of the problem

How can **a multi-task model** perform <strong class="classification">humanitarian classification</strong> and <strong class="ner">NER</strong> on Taglish typhoon posts while keeping **negative transfer** and **training cost** low?

</div>

---

<div class="quote">

### General objective

The general objective is to **develop a multi-task model** that performs <strong class="classification">humanitarian classification</strong> and <strong class="ner">NER</strong> on Taglish typhoon posts.

</div>

---
clicks: 4
---

<SpecificObjectives />

---
clicks: 10
---

# Baseline studies and the gap

<LiteratureTable />

---
clicks: 8
---

<DatasetSplit />

---
clicks: 4
---

<LabelCheck />

---
clicks: 7
---

<MeasuresPanel />

---
clicks: 5
---

<OptimizerGates />

---
clicks: 3
---

<FinalComparison />

---
clicks: 2
---

<PrototypeTests />

---
clicks: 6
---

<ScheduleGantt />

---
clicks: 3
---

<NextSteps />

---
clicks: 3
---

<LabelingMilestones />

---
clicks: 5
---

<RiskCards />

---

<div class="quote">

### In summary

One model that <strong class="classification">sorts Taglish typhoon posts by type of report</strong> and <strong class="ner">picks out their key details</strong>, with the optimizer that best reduces **negative transfer** without needing too much training time or memory

</div>

---

<ReferenceList :page="1" />

---

<ReferenceList :page="2" />
