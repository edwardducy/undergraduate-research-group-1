<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { draw, enter, exit, moveShort, pop } from '../shared/motion'

// Placeholder results until training is done. One classification F1-score per seed for each model,
// and the five NER gaps (final model minus single-task).
const single = [0.720, 0.701, 0.739, 0.708, 0.731]
const multi = [0.735, 0.709, 0.747, 0.720, 0.742]
const nerGaps = [0.006, -0.012, 0.004, -0.009, -0.008]

// The paired chart on the left.
const chart = { top: 100, bottom: 404, low: 0.68, high: 0.76 }
const y = (score: number) => chart.bottom - (score - chart.low) * (chart.bottom - chart.top) / (chart.high - chart.low)
const x = { single: 110, multi: 300 }

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length
// The mean gap and the half-width of its 95% confidence interval. 2.776 is the t value for 4 degrees of freedom.
const interval = (gaps: number[]) => {
  const centre = mean(gaps)
  const spread = Math.sqrt(gaps.reduce((sum, gap) => sum + (gap - centre) ** 2, 0) / (gaps.length - 1))
  return { centre, half: 2.776 * spread / Math.sqrt(gaps.length) }
}

const runs = single.map((score, run) => ({
  run,
  y1: y(score),
  y2: y(multi[run]),
  gap: multi[run] - score,
  length: Math.hypot(x.multi - x.single, y(multi[run]) - y(score)),
}))

// The gap line on the right. 0 sits in the middle, and the final model is better to the right.
const line = { left: 440, right: 884, perPoint: 9000 }
const zero = (line.left + line.right) / 2
const gapX = (gap: number) => zero + gap * line.perPoint

const rows = [
  { key: 'classification', name: 'Classification', y: 180, gaps: runs.map(run => run.gap) },
  { key: 'ner', name: 'NER', y: 300, gaps: nerGaps },
].map((row) => {
  const { centre, half } = interval(row.gaps)
  const clear = centre - half > 0 || centre + half < 0
  return {
    ...row,
    centre: gapX(centre),
    from: gapX(centre - half),
    to: gapX(centre + half),
    verdict: clear ? 'Clear difference' : 'No clear difference',
    clear,
  }
})

const sign = (gap: number) => `${gap >= 0 ? '+' : '−'}${Math.abs(gap).toFixed(3)}`
const faded = 0.3
const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.runs, .gap-label, .scale, .row, .average, .average-label, .test-text, .interval, .interval-dot, .verdict, .interval-note, .placeholder', { opacity: 0 })
  gsap.set('.link', { strokeDashoffset: (_: number, element: Element) => Number(element.getAttribute('data-length')) })
  gsap.set('.average', { scale: 0, transformOrigin: '50% 50%' })
  gsap.set('.interval', { scaleX: 0, transformOrigin: '50% 50%' })
  runs.forEach(run => gsap.set(`.row-0 .gap-dot[data-run="${run.run}"]`, { attr: { cx: x.multi, cy: run.y2 } }))

  timeline.addLabel('step0')

  // Click 1. Each seed gives one run of each model. A line joins the two runs, and the gap is written beside it.
  timeline
    .to('.pill', { opacity: faded, ...exit })
    .to('.runs, .placeholder', { opacity: 1, duration: 0 }, '<')
    .fromTo('.single-dot', { opacity: 0 }, { opacity: 1, ...enter, stagger: 0.05 }, '<')
  runs.forEach((run) => {
    timeline
      .to(`.link[data-run="${run.run}"]`, { strokeDashoffset: 0, ...draw }, run.run ? '<0.12' : '>')
      .fromTo(`.multi-dot[data-run="${run.run}"]`, { opacity: 0 }, { opacity: 1, ...enter }, '>-0.15')
      .to(`.gap-label[data-run="${run.run}"]`, { opacity: 1, ...enter }, '<')
  })
  timeline.addLabel('step1')

  // Click 2. Each pair becomes one gap on the number line, and the average gap is marked.
    .to('.paired', { opacity: faded, ...exit })
    .to('.gap-label', { opacity: 0, ...exit }, '<')
    .to('.scale, .row-0', { opacity: 1, ...enter }, '<0.1')
  runs.forEach((run) => {
    timeline.to(`.row-0 .gap-dot[data-run="${run.run}"]`, { attr: { cx: gapX(run.gap), cy: rows[0].y }, ...moveShort }, run.run ? '<0.08' : '>')
  })
  timeline
    .to('.row-0 .average', { opacity: 1, scale: 1, ...pop }, '>')
    .to('.average-label, .test-text', { opacity: 1, ...enter }, '<0.1')
    .addLabel('step2')

  // Click 3. Each bar shows the 95% confidence interval. A bar that does not touch 0 means a clear difference.
    .to('.average-label, .test-text', { opacity: 0, ...exit })
    .to('.row-0 .average', { opacity: 0, duration: 0 }, '>')
    .to('.row-0 .interval, .row-0 .interval-dot', { opacity: 1, duration: 0 }, '<')
    .to('.row-0 .interval', { scaleX: 1, ...draw }, '<')
    .to('.row-0 .verdict, .interval-note', { opacity: 1, ...enter }, '>-0.1')
    .to('.row-1', { opacity: 1, ...enter }, '>0.2')
    .to('.row-1 .interval, .row-1 .interval-dot', { opacity: 1, duration: 0 }, '>')
    .to('.row-1 .interval', { scaleX: 1, ...draw }, '<')
    .to('.row-1 .verdict', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step3')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <defs>
      <marker id="comparison-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">
        <path class="head" d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
    </defs>

    <g class="paired">
      <g class="pill">
        <rect class="pill-box" :x="(x.single + x.multi) / 2 - 56" y="0" width="112" height="28" rx="14" />
        <text class="pill-text" :x="(x.single + x.multi) / 2" y="19" text-anchor="middle">Test posts</text>
      </g>
      <text class="column-name" :x="x.single" y="60" text-anchor="middle">Single-task</text>
      <text class="column-name" :x="x.multi" y="60" text-anchor="middle">Final model</text>
      <text class="axis-name" x="0" y="88">F1-score</text>
      <path class="column-line" :d="`M ${x.single} ${chart.top} V ${chart.bottom} M ${x.multi} ${chart.top} V ${chart.bottom}`" />

      <g class="runs">
        <g v-for="run in runs" :key="run.run">
          <path class="link" :data-run="run.run" :data-length="run.length" :style="{ strokeDasharray: run.length }" :d="`M ${x.single} ${run.y1} L ${x.multi} ${run.y2}`" />
          <circle class="single-dot" :cx="x.single" :cy="run.y1" r="7" />
          <circle class="multi-dot" :data-run="run.run" :cx="x.multi" :cy="run.y2" r="7" />
          <text class="seed" :x="x.single - 16" :y="run.y1 + 5" text-anchor="end">Seed {{ run.run + 1 }}</text>
        </g>
      </g>
    </g>
    <text v-for="run in runs" :key="`gap-${run.run}`" class="gap-label" :data-run="run.run" :x="x.multi + 16" :y="run.y2 + 5">{{ sign(run.gap) }}</text>

    <g class="scale">
      <path class="zero" :d="`M ${zero} 90 V 380`" />
      <text class="zero-text" :x="zero" y="402" text-anchor="middle">0</text>
      <text class="side" :x="zero" y="420" text-anchor="middle">no difference</text>
      <text class="side strong" :x="zero + 14" y="70">Final model better</text>
      <path class="side-arrow" :d="`M ${line.right - 60} 66 H ${line.right}`" marker-end="url(#comparison-head)" />
      <text class="side strong" :x="zero - 14" y="70" text-anchor="end">Single-task better</text>
      <path class="side-arrow" :d="`M ${line.left + 60} 66 H ${line.left}`" marker-end="url(#comparison-head)" />
    </g>

    <g v-for="(row, index) in rows" :key="row.key" class="row" :class="`row-${index}`">
      <text class="task" :class="row.key" :x="line.left" :y="row.y - 34">{{ row.name }}</text>
      <path class="gap-line" :d="`M ${line.left} ${row.y} H ${line.right}`" />
      <circle v-for="(gap, run) in row.gaps" :key="run" class="gap-dot" :data-run="run" :cx="gapX(gap)" :cy="row.y" r="6" />
      <path
        class="interval" :class="row.key"
        :d="`M ${row.from} ${row.y} H ${row.to} M ${row.from} ${row.y - 10} V ${row.y + 10} M ${row.to} ${row.y - 10} V ${row.y + 10}`"
        :style="{ transformOrigin: `${row.centre}px ${row.y}px` }"
      />
      <circle class="average" :class="row.key" :cx="row.centre" :cy="row.y" r="10" />
      <circle class="interval-dot" :class="row.key" :cx="row.centre" :cy="row.y" r="10" />
      <text class="verdict" :class="{ clear: row.clear }" :x="line.right" :y="row.y + 36" text-anchor="end">{{ row.verdict }}</text>
    </g>
    <text class="average-label" :x="rows[0].centre" :y="rows[0].y - 18" text-anchor="middle">Average gap</text>

    <text class="test-text title" :x="line.left" y="432">Paired t-test</text>
    <text class="test-text" :x="line.left" y="452">Is the average gap clearly away from 0?</text>
    <text class="interval-note" :x="line.left" y="446">Each bar shows the 95% confidence interval</text>

    <text class="placeholder" x="0" y="452">Placeholder values until training is done</text>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.head {
  fill: var(--color-subtle);
}

.pill-box {
  fill: var(--color-land);
}

.pill-text {
  font-size: 0.9rem;
  font-weight: 600;
  fill: var(--color-ink);
}

.column-name {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.axis-name {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.column-line {
  stroke: var(--color-faint);
  stroke-width: 2;
}

.link {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.single-dot {
  fill: var(--color-subtle);
}

.multi-dot {
  fill: var(--color-ink);
}

.seed {
  font-size: 0.75rem;
  fill: var(--color-subtle);
}

.gap-label {
  font-size: 0.8rem;
  font-weight: 700;
  fill: var(--color-positive);
}

.zero {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 1.5;
  stroke-dasharray: 5 5;
}

.zero-text {
  font-size: 0.9rem;
  font-weight: 700;
  fill: var(--color-muted);
}

.side {
  font-size: 0.8rem;
  fill: var(--color-subtle);
}

.side.strong {
  font-weight: 600;
}

.side-arrow {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.5;
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

.gap-line {
  stroke: var(--color-faint);
  stroke-width: 2;
}

.gap-dot {
  fill: var(--color-faint);
  stroke: var(--color-surface);
  stroke-width: 1.5;
}

.interval {
  fill: none;
  stroke-width: 5;
  stroke-linecap: round;
}

.interval.classification {
  stroke: var(--color-classification);
}

.interval.ner {
  stroke: var(--color-ner);
}

.average.classification,
.interval-dot.classification {
  fill: var(--color-classification);
  stroke: none;
}

.average.ner,
.interval-dot.ner {
  fill: var(--color-ner);
  stroke: none;
}

.average-label {
  font-size: 0.8rem;
  font-weight: 700;
  fill: var(--color-classification);
}

.verdict {
  font-size: 0.85rem;
  font-weight: 700;
  fill: var(--color-subtle);
}

.verdict.clear {
  fill: var(--color-positive);
}

.test-text {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.test-text.title {
  font-size: 0.95rem;
  font-weight: 700;
  fill: var(--color-ink);
}

.interval-note {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.placeholder {
  font-size: 0.75rem;
  fill: var(--color-subtle);
}
</style>
