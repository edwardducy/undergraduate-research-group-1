<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Laid out like the optimizer slide. The steps sit on the left, and a paired estimation plot
// (Gardner and Altman) builds up on the right. One line joins the two runs that share a seed,
// then the mean difference and its 95% confidence interval go on their own axis

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

// Lines fade back but stay readable once the means take over
const dim = 0.3

const steps = [
  { name: 'Same test posts', question: 'Final model vs single-task models' },
  { name: 'Five random seeds', question: 'Every run has a matching run' },
  { name: 'Paired t-test', question: 'Do the mean F1-scores differ?' },
  { name: '95% confidence interval', question: 'The likely size of each difference' },
]

// Columns for the single-task runs, the multi-task runs and the differences
const x = { single: 56, multi: 246, difference: 390 }
const panelHeight = 116
// The difference axis runs from -.04 to +.04
const range = 0.04

// Made-up F1-scores for five runs. Run k of both models uses seed k
const tasks = [
  { key: 'classification', name: 'Classification', top: 150, low: 0.695, high: 0.755, single: [0.72, 0.70, 0.74, 0.71, 0.73], multi: [0.73, 0.71, 0.75, 0.70, 0.74] },
  { key: 'ner', name: 'NER', top: 316, low: 0.565, high: 0.635, single: [0.61, 0.60, 0.63, 0.59, 0.62], multi: [0.58, 0.59, 0.60, 0.57, 0.61] },
].map((task) => {
  const y = (value: number) => task.top + panelHeight * (task.high - value) / (task.high - task.low)
  const yDifference = (value: number) => task.top + panelHeight / 2 - (panelHeight / 2) * value / range
  const differences = task.multi.map((score, run) => score - task.single[run])
  const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length
  const spread = Math.sqrt(differences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (differences.length - 1))
  // The paired t-test's 95% confidence interval, with the t value for four degrees of freedom
  const margin = 2.776 * spread / Math.sqrt(differences.length)
  const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length
  const pairs = task.single.map((score, run) => {
    const y1 = y(score)
    const y2 = y(task.multi[run])
    return { run, y1, y2, higher: task.multi[run] > score, length: Math.hypot(x.multi - x.single, y2 - y1) }
  })
  return {
    ...task,
    pairs,
    meanSingle: y(average(task.single)),
    meanMulti: y(average(task.multi)),
    zero: yDifference(0),
    dots: differences.map(yDifference),
    mean: yDifference(mean),
    ciHigh: yDifference(mean + margin),
    ciLow: yDifference(mean - margin),
  }
})

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  // Step 1 is on from the start, the others wait in a faint grey
  gsap.set('.item, .item .question', { color: colour('faint') })
  gsap.set('.item-0', { color: colour('ink') })
  gsap.set('.item-0 .question', { color: colour('muted') })
  gsap.set('.pair .link', { strokeDashoffset: (_: number, element: Element) => Number(element.getAttribute('data-length')) })
  gsap.set('.pair circle, .difference-axis, .difference-dot, .note', { opacity: 0 })
  gsap.set('.mean-mark', { scaleX: 0, transformOrigin: '50% 50%' })
  gsap.set('.mean-difference', { scale: 0, transformOrigin: '50% 50%' })
  gsap.set('.interval', { scaleY: 0, transformOrigin: '50% 50%' })

  let previous = 0
  const focus = (index: number) => {
    timeline
      .to(`.item-${previous}, .item-${previous} .question`, { color: colour('faint'), ...exit })
      .to(`.item-${index}`, { color: colour('ink'), ...enter }, '<')
      .to(`.item-${index} .question`, { color: colour('muted'), ...enter }, '<')
    previous = index
  }

  timeline.addLabel('step0')

  // Five seeds. Each line joins the two runs that started the same way
  focus(1)
  timeline
    .to('.pair .single-dot', { opacity: 1, ...enter }, '<')
  for (let run = 0; run < 5; run++) {
    timeline
      .to(`.pair[data-run="${run}"] .link`, { strokeDashoffset: 0, ...draw }, run === 0 ? '>' : '<0.12')
      .to(`.pair[data-run="${run}"] .multi-dot`, { opacity: 1, ...enter }, '>-0.15')
  }
  timeline
    .to('.note.pairs', { opacity: 1, ...enter }, '<')
    .addLabel('step1')

  // The paired t-test. The means of each column, and the gap between them on its own axis
  focus(2)
  timeline
    .to('.note.pairs', { opacity: 0, ...exit }, '<')
    .to('.pair', { opacity: dim, ...exit }, '<')
    .to('.mean-mark', { scaleX: 1, ...draw }, '>')
    .to('.difference-axis', { opacity: 1, ...enter }, '<')
    .to('.difference-dot', { opacity: 1, ...enter, stagger: 0.05 }, '>')
    .to('.mean-difference', { scale: 1, ...pop }, '>')
    .to('.note.means', { opacity: 1, ...enter }, '<')
    .addLabel('step2')

  // The 95% confidence interval grows around each mean difference
  focus(3)
  timeline
    .to('.note.means', { opacity: 0, ...exit }, '<')
    .to('.interval', { scaleY: 1, ...draw }, '<0.1')
    .to('.note.interval-note', { opacity: 1, ...enter }, '<')
    .addLabel('step3')
})
</script>

<template>
  <div ref="root" class="comparison">
    <section>
      <h3>Comparing the models</h3>
      <ol>
        <li v-for="(step, index) in steps" :key="step.name" class="item" :class="`item-${index}`">
          <b>{{ index + 1 }}</b>
          <div>
            <p class="name">{{ step.name }}</p>
            <p class="question">{{ step.question }}</p>
          </div>
        </li>
      </ol>
    </section>

    <svg viewBox="0 0 440 456" width="440" height="456">
      <!-- The two models head their columns, the same boxes as on the model slide -->
      <g :transform="`translate(${x.single} 0)`">
        <rect class="encoder" x="-60" y="0" width="56" height="24" rx="5" />
        <rect class="encoder" x="4" y="0" width="56" height="24" rx="5" />
        <rect class="head classification" x="-60" y="30" width="56" height="16" rx="4" />
        <rect class="head ner" x="4" y="30" width="56" height="16" rx="4" />
        <text class="column-name" y="74" text-anchor="middle">Single-task</text>
      </g>
      <g :transform="`translate(${x.multi} 0)`">
        <rect class="encoder" x="-50" y="0" width="100" height="24" rx="5" />
        <rect class="head classification" x="-50" y="30" width="46" height="16" rx="4" />
        <rect class="head ner" x="4" y="30" width="46" height="16" rx="4" />
        <text class="column-name" y="74" text-anchor="middle">Multi-task</text>
      </g>
      <rect class="pill" :x="(x.single + x.multi) / 2 - 52" y="88" width="104" height="24" rx="12" />
      <text class="pill-text" :x="(x.single + x.multi) / 2" y="105" text-anchor="middle">Test posts</text>
      <text class="column-name difference-axis" :x="x.difference" y="74" text-anchor="middle">Difference</text>

      <g v-for="task in tasks" :key="task.key">
        <text class="task" :class="task.key" x="0" :y="task.top - 12">{{ task.name }}</text>
        <path class="column-line" :d="`M ${x.single} ${task.top} V ${task.top + panelHeight} M ${x.multi} ${task.top} V ${task.top + panelHeight}`" />

        <!-- One line for each seed. Dark if the multi-task run scored higher, red if lower -->
        <g v-for="pair in task.pairs" :key="pair.run" class="pair" :data-run="pair.run">
          <path
            class="link" :class="{ lower: !pair.higher }" :d="`M ${x.single} ${pair.y1} L ${x.multi} ${pair.y2}`"
            :data-length="pair.length" :style="{ strokeDasharray: pair.length }"
          />
          <circle class="single-dot" :cx="x.single" :cy="pair.y1" r="6" />
          <circle class="multi-dot" :cx="x.multi" :cy="pair.y2" r="6" />
        </g>

        <path class="mean-mark" :class="task.key" :d="`M ${x.single - 18} ${task.meanSingle} H ${x.single + 18}`" />
        <path class="mean-mark" :class="task.key" :d="`M ${x.multi - 18} ${task.meanMulti} H ${x.multi + 18}`" />

        <!-- The difference axis, with 0 where the two models score the same -->
        <g class="difference-axis">
          <path class="column-line dark" :d="`M ${x.difference} ${task.top} V ${task.top + panelHeight}`" />
          <path class="zero" :d="`M ${x.difference - 22} ${task.zero} H ${x.difference + 22}`" />
          <text class="zero-text" :x="x.difference + 28" :y="task.zero + 5">0</text>
        </g>
        <circle v-for="(dot, run) in task.dots" :key="run" class="difference-dot" :cx="x.difference - 30" :cy="dot" r="4" />
        <path
          class="interval" :class="task.key"
          :d="`M ${x.difference} ${task.ciHigh} V ${task.ciLow} M ${x.difference - 9} ${task.ciHigh} H ${x.difference + 9} M ${x.difference - 9} ${task.ciLow} H ${x.difference + 9}`"
          :style="{ transformOrigin: `${x.difference}px ${task.mean}px` }"
        />
        <circle class="mean-difference" :class="task.key" :cx="x.difference" :cy="task.mean" r="8" />
      </g>

      <text class="note pairs" :x="(x.single + x.multi) / 2" y="452" text-anchor="middle">One line for each matching pair of runs</text>
      <text class="note means" x="440" y="452" text-anchor="end">Mean difference, for each task</text>
      <text class="note interval-note" x="440" y="452" text-anchor="end">Bar shows the 95% confidence interval</text>
    </svg>

    <footer>Illustration, not results</footer>
  </div>
</template>

<style scoped>
.comparison {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 456px;
}

section {
  flex: none;
  width: 390px;
}

h3 {
  margin-bottom: var(--space-2);
}

ol {
  display: grid;
  gap: var(--space-2);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: grid;
  grid-template-columns: 2rem 1fr;
  line-height: var(--leading-list);
}

li b {
  font-weight: 700;
}

.name {
  margin: 0;
  font-weight: 700;
  line-height: var(--leading-list);
}

.question {
  margin: 0;
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
}

svg {
  flex: none;
  overflow: visible;
}

.encoder {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head.classification {
  fill: color-mix(in srgb, var(--color-classification) 20%, var(--color-surface));
  stroke: var(--color-classification);
  stroke-width: 1.5;
}

.head.ner {
  fill: color-mix(in srgb, var(--color-ner) 20%, var(--color-surface));
  stroke: var(--color-ner);
  stroke-width: 1.5;
}

.column-name {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.pill {
  fill: var(--color-land);
}

.pill-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.task {
  font-size: var(--text-xs);
  font-weight: 700;
}

.task.classification {
  fill: var(--color-classification);
}

.task.ner {
  fill: var(--color-ner);
}

.column-line {
  stroke: var(--color-faint);
  stroke-width: 2;
}

.column-line.dark {
  stroke: var(--color-subtle);
}

.link {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.link.lower {
  stroke: var(--color-negative);
}

.single-dot {
  fill: var(--color-subtle);
}

.multi-dot {
  fill: var(--color-ink);
}

.mean-mark,
.interval {
  fill: none;
  stroke-width: 5;
  stroke-linecap: round;
}

.interval {
  stroke-width: 4;
}

.mean-mark.classification,
.interval.classification {
  stroke: var(--color-classification);
}

.mean-mark.ner,
.interval.ner {
  stroke: var(--color-ner);
}

.mean-difference.classification {
  fill: var(--color-classification);
}

.mean-difference.ner {
  fill: var(--color-ner);
}

.zero {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 4 4;
}

.zero-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.difference-dot {
  fill: var(--color-faint);
}

.note {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

footer {
  position: absolute;
  left: 0;
  bottom: 0;
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
