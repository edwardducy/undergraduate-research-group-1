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

# The dataset and its three parts

<DatasetSplit />

---
clicks: 4
---

# Labels are double-checked

<LabelCheck />

---
clicks: 5
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

# The prototype is checked feature by feature

<PrototypeTests />

---
clicks: 6
---

# The plan, November 2026 to April 2027

<ScheduleGantt />

---
clicks: 3
---

# Three reasons the plan is realistic

<FeasibilityCards />

---
clicks: 3
---

# Immediate next steps

<NextSteps />

---
clicks: 4
---

# Three checkpoints before full labeling

<LabelingGate />

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

---

<ReferenceList :page="3" />

---

<ReferenceList :page="4" />

---

<ReferenceList :page="5" />

---

<ReferenceList :page="6" />
