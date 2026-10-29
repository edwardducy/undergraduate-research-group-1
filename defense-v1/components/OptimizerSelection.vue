<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, pop } from '../shared/motion'

const dim = 0.3

const steps = [
  { name: 'Analysis of variance', question: 'Do the 13 differ at all?' },
  { name: 'Dunnett\'s test', question: 'Clearly less than equal weighting?' },
  { name: 'Both task scores', question: 'Neither lower than equal weighting?' },
  { name: 'Training time', question: 'The fastest is the final optimizer' },
  { name: 'If none qualifies', question: 'Keep equal weighting, at no extra cost' },
]

const chart = { left: 30, width: 410, top: 104, base: 368 }
const span = chart.base - chart.top
const heights = [0.60, 0.70, 0.35, 0.55, 0.20, 0.80, 0.58, 0.30, 0.65, 0.62, 0.25, 0.50, 0.90]
const mean = heights.reduce((sum, value) => sum + value, 0) / heights.length
type Passed = { index: number, lowers: 'classification' | 'ner' } | { index: number, time: number }
const dunnett: Passed[] = [
  { index: 2, time: 0.70 },
  { index: 4, lowers: 'ner' },
  { index: 7, time: 0.40 },
  { index: 10, time: 0.85 },
]
const passed = dunnett.map(item => item.index)
const lowers: Record<number, string> = Object.fromEntries(dunnett.flatMap(item => 'lowers' in item ? [[item.index, item.lowers]] : []))
const qualified = dunnett.flatMap(item => 'time' in item ? [item] : [])
const time: Record<number, number> = Object.fromEntries(qualified.map(item => [item.index, item.time]))
const fastest = qualified.reduce((a, b) => (a.time < b.time ? a : b)).index

const columns = heights.map((height, index) => ({
  index,
  x: chart.left + (chart.width / heights.length) * (index + 0.5),
  height: height * span,
  time: (time[index] ?? 0) * span,
}))
const ewY = chart.base - heights[0] * span
const at = (selector: string, indexes: number[]) => indexes.map(index => `${selector}[data-index="${index}"]`).join(', ')
const others = columns.slice(1).map(column => column.index)

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.item, .item .question', { color: colour('faint') })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.time-bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.mean-line, .note, .ew-line, .ew-label, .badge, .task-key, .time-legend, .final-label', { opacity: 0 })
  gsap.set('.badge', { scale: 0.4, transformOrigin: '50% 50%' })

  let previous = -1
  const focus = (index: number) => {
    if (previous >= 0)
      timeline.to(`.item-${previous}, .item-${previous} .question`, { color: colour('faint'), ...exit })
    timeline
      .to(`.item-${index}`, { color: colour('ink'), ...enter }, previous >= 0 ? '<' : '>')
      .to(`.item-${index} .question`, { color: colour('muted'), ...enter }, '<')
    previous = index
  }

  timeline.addLabel('step0')

  focus(0)
  timeline
    .to('.bar', { scaleY: (index: number) => mean / heights[index], ...draw }, '<')
    .to('.bar', { scaleY: 1, ...draw, stagger: 0.03 }, '>0.2')
    .to('.mean-line, .note.differ', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step1')

  focus(1)
  timeline
    .to('.mean-line, .note.differ', { opacity: 0, ...exit }, '<')
    .to('.bar[data-index="0"]', { fill: colour('ink'), ...enter }, '<')
    .to('.ew-line, .ew-label', { opacity: 1, ...enter }, '<')
    .to(at('.column', others.filter(index => !passed.includes(index))), { opacity: 0, ...exit, duration: 0.4 }, '<0.2')
    .to(at('.badge.pass', passed), { opacity: 1, scale: 1, ...pop, stagger: 0.12 }, '>')
    .to('.note.below', { opacity: 1, ...enter }, '<')
    .addLabel('step2')

  focus(2)
  timeline
    .to('.badge.pass, .note.below', { opacity: 0, ...exit }, '<')
    .to('.task-key', { opacity: 1, ...enter }, '<')
    .to('.badge.task', { opacity: 1, scale: 1, ...pop, stagger: 0.08 }, '>')
    .to(at('.column', Object.keys(lowers).map(Number)), { opacity: dim, ...exit }, '>0.2')
    .to('.note.both', { opacity: 1, ...enter }, '<')
    .addLabel('step3')

  focus(3)
  timeline
    .to('.badge.task, .task-key, .note.both, .ew-line, .ew-label, .loss-legend', { opacity: 0, ...exit }, '<')
    .to('.bar', { scaleY: 0, ...exit }, '<')
    .to(at('.column', Object.keys(lowers).map(Number)), { opacity: 0, ...exit }, '<')
    .to('.column[data-index="0"]', { opacity: dim, ...exit }, '<')
    .to('.time-legend', { opacity: 1, ...enter }, '>')
    .to('.time-bar', { scaleY: 1, ...draw, stagger: 0.08 }, '<')
    .to(`.time-bar[data-index="${fastest}"]`, { fill: colour('positive'), ...enter }, '>')
    .to('.final-label, .note.qualify', { opacity: 1, ...enter }, '<')
    .addLabel('step4')

  focus(4)
  timeline
    .to('.time-bar, .final-label, .note.qualify, .time-legend', { opacity: 0, ...exit }, '<')
    .to(at('.column', others), { opacity: 0, ...exit }, '<')
    .to('.loss-legend', { opacity: 1, ...enter }, '>')
    .to('.bar[data-index="0"]', { scaleY: 1, ...draw }, '<')
    .to('.column[data-index="0"]', { opacity: 1, ...enter }, '<')
    .to('.bar[data-index="0"]', { fill: colour('positive'), ...enter }, '<')
    .to('.note.kept', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="choice">
    <section>
      <h3>Choosing the optimizer</h3>
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
      <g class="loss-legend">
        <rect class="swatch loss" :x="chart.left" :y="chart.top - 50" width="14" height="14" rx="3" />
        <text class="tag" :x="chart.left + 22" :y="chart.top - 37">Negative transfer</text>
      </g>
      <g class="time-legend">
        <rect class="swatch time" :x="chart.left" :y="chart.top - 50" width="14" height="14" rx="3" />
        <text class="tag" :x="chart.left + 22" :y="chart.top - 37">Training time</text>
      </g>
      <g class="task-key">
        <circle class="classification" :cx="chart.left + 266" :cy="chart.top - 43" r="7" />
        <text class="key-text" :x="chart.left + 278" :y="chart.top - 37">Classification</text>
        <circle class="ner" :cx="chart.left + 266" :cy="chart.top - 19" r="7" />
        <text class="key-text" :x="chart.left + 278" :y="chart.top - 13">NER</text>
      </g>

      <path class="axis" :d="`M ${chart.left} ${chart.top} V ${chart.base} H ${chart.left + chart.width}`" />
      <text class="tag" text-anchor="middle" :transform="`translate(${chart.left - 16} ${(chart.top + chart.base) / 2}) rotate(-90)`">Lower is better</text>

      <g v-for="column in columns" :key="column.index" class="column" :data-index="column.index">
        <rect class="bar" :data-index="column.index" :x="column.x - 10" :y="chart.base - column.height" width="20" :height="column.height" />
        <rect v-if="column.time" class="time-bar" :data-index="column.index" :x="column.x - 10" :y="chart.base - column.time" width="20" :height="column.time" />
        <text class="number" :x="column.x" :y="chart.base + 28" text-anchor="middle">{{ column.index + 1 }}</text>
      </g>

      <g v-for="index in passed" :key="index">
        <g class="badge pass" :data-index="index" :transform="`translate(${columns[index].x} ${chart.base - columns[index].height - 22})`">
          <circle class="ok" r="11" />
          <path class="tick" d="M -5 0 L -1.5 4 L 5.5 -4" />
        </g>
        <g v-for="(task, slot) in ['classification', 'ner']" :key="task" class="badge task" :transform="`translate(${columns[index].x} ${chart.base - columns[index].height - 22 - slot * 26})`">
          <circle :class="lowers[index] === task ? 'fail' : task" r="11" />
          <path v-if="lowers[index] === task" class="tick" d="M -4.5 -4.5 L 4.5 4.5 M 4.5 -4.5 L -4.5 4.5" />
          <path v-else class="tick" d="M -5 0 L -1.5 4 L 5.5 -4" />
        </g>
      </g>

      <path class="mean-line" :d="`M ${chart.left} ${chart.base - mean * span} H ${chart.left + chart.width}`" />
      <path class="ew-line" :d="`M ${chart.left} ${ewY} H ${chart.left + chart.width}`" />
      <text class="final-label" :x="columns[fastest].x" :y="chart.base - columns[fastest].time - 16" text-anchor="middle">Final optimizer</text>

      <text class="note-text ew-label" :x="columns[0].x - 14" :y="chart.base + 64">Equal weighting</text>
      <text class="note-text note differ" :x="chart.left + chart.width" :y="chart.base + 64" text-anchor="end">Not all at one height, so they differ</text>
      <text class="note-text note below good" :x="chart.left + chart.width" :y="chart.base + 64" text-anchor="end">{{ passed.length }} clearly below its line</text>
      <text class="note-text note both good" :x="chart.left + chart.width" :y="chart.base + 64" text-anchor="end">{{ qualified.length }} keep both task scores</text>
      <text class="note-text note qualify" :x="chart.left + chart.width" :y="chart.base + 64" text-anchor="end">Only the {{ qualified.length }} that qualify</text>
      <text class="note-text note kept good" :x="columns[0].x - 14" :y="chart.base + 64">Equal weighting is kept, at no extra cost</text>
    </svg>

  </div>
</template>

<style scoped>
.choice {
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
  gap: var(--space-1);
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

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.key-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.axis {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.bar,
.swatch.loss {
  fill: var(--color-negative);
}

.time-bar,
.swatch.time {
  fill: var(--color-ink);
}

.number {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.mean-line {
  stroke: var(--color-subtle);
  stroke-width: 2;
  stroke-dasharray: 4 5;
}

.ew-line {
  stroke: var(--color-ink);
  stroke-width: 2.5;
  stroke-dasharray: 8 6;
}

.badge .ok {
  fill: var(--color-positive);
}

.badge .classification,
circle.classification {
  fill: var(--color-classification);
}

.badge .ner,
circle.ner {
  fill: var(--color-ner);
}

.badge .fail {
  fill: var(--color-negative);
}

.tick {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.note-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.note-text.good,
.final-label {
  fill: var(--color-positive);
}

.final-label {
  font-size: var(--text-2xs);
  font-weight: 700;
}

</style>
