<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, pop } from '../shared/motion'

const reasons = [
  { name: 'Training pilot', detail: 'Every model fits in memory', x: 0 },
  { name: 'Fixed workload', detail: '143 runs, known in advance', x: 484 },
]
const divider = 442
const head = { y: 45, radius: 15 }
const faded = 0.3

// Reason 1. Placeholder memory after freezing, the same numbers as the risks slide.
const memory = [6.1, 7.2, 6.8, 9.6, 7.5, 6.4, 10.3, 7.0, 6.6, 11.1, 7.7, 9.2, 6.9].map(gb => gb * 0.68)
const chart = { left: 20, right: 390, base: 360, perGb: 24 }
const limit = chart.base - 8 * chart.perGb
const slot = (chart.right - chart.left) / memory.length
const bars = memory.map((gb, index) => ({ x: chart.left + index * slot + slot * 0.2, width: slot * 0.6, height: gb * chart.perGb }))

// Reason 2. One square for each run.
const grid = { x: reasons[1].x + 10, columns: 16, pitch: 15, size: 11 }
const squares = (count: number, top: number) => Array.from({ length: count }, (_, index) => ({
  x: grid.x + (index % grid.columns) * grid.pitch,
  y: top + Math.floor(index / grid.columns) * grid.pitch,
}))
const tuning = squares(92, 115)
const main = squares(51, 240)
// A quarter of a month is 20 units. Training runs from late January to late February.
const months = { x: grid.x, quarter: 20, axis: 388 }
const training = { x: months.x + 3 * months.quarter, width: 4.5 * months.quarter }

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  const looks = {
    neutral: { name: 'ink', ring: 'ink', fill: 'surface', number: 'ink', shown: 1, tick: 0, detail: 0 },
    off: { name: 'faint', ring: 'faint', fill: 'surface', number: 'faint', shown: 1, tick: 0, detail: 0 },
    on: { name: 'ink', ring: 'ink', fill: 'ink', number: 'surface', shown: 1, tick: 0, detail: 1 },
    done: { name: 'subtle', ring: 'subtle', fill: 'surface', number: 'subtle', shown: 0, tick: 1, detail: 1 },
  }
  type Look = keyof typeof looks

  const style = (index: number, name: Look) => {
    const look = looks[name]
    const item = `.reason-${index}`
    timeline
      .to(`${item} .name`, { fill: colour(look.name), ...enter }, '<')
      .to(`${item} .ring`, { stroke: colour(look.ring), fill: colour(look.fill), ...enter }, '<')
      .to(`${item} .number`, { fill: colour(look.number), opacity: look.shown, ...enter }, '<')
      .to(`${item} .tick`, { opacity: look.tick, ...enter }, '<')
      .to(`${item} .detail`, { opacity: look.detail, ...enter }, '<')
  }

  // Lights up one reason, marks earlier ones as done, and fades the column before it.
  const focus = (reason: number) => {
    if (reason)
      timeline.to(`.column-${reason - 1}`, { opacity: faded, ...exit })
    reasons.forEach((_, index) => style(index, index < reason ? 'done' : index === reason ? 'on' : 'off'))
    timeline.to(`.column-${reason}`, { opacity: 1, ...enter }, '<0.1')
  }

  reasons.forEach((_, index) => {
    const look = looks.neutral
    gsap.set(`.reason-${index} .name`, { fill: colour(look.name) })
    gsap.set(`.reason-${index} .ring`, { stroke: colour(look.ring), fill: colour(look.fill) })
    gsap.set(`.reason-${index} .number`, { fill: colour(look.number) })
  })
  gsap.set('.tick, .detail, .column, .reveal', { opacity: 0 })
  gsap.set('.check, .pill', { scale: 0.4, transformOrigin: '50% 50%' })
  gsap.set('.memory-bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.training', { scaleX: 0, transformOrigin: '0% 50%' })

  timeline.addLabel('step0')

  // Click 1. Every memory bar stays under the 8 GB line after one change.
  focus(0)
  timeline
    .to('.memory-bar', { scaleY: 1, ...draw, stagger: 0.03 }, '<0.1')
    .to('.pill', { opacity: 1, scale: 1, ...pop }, '>-0.1')
    .to('.fit', { opacity: 1, ...enter }, '<0.15')
    .to('.fit .check', { scale: 1, ...pop }, '<')
    .addLabel('step1')

  // Click 2. One square appears for each tuning run, then for each main run.
  focus(1)
  timeline
    .to('.tuning', { opacity: 1, ...enter, stagger: 0.008 }, '>')
    .to('.main', { opacity: 1, ...enter, stagger: 0.008 }, '>-0.1')
    .addLabel('step2')

  // Click 3. The runs lead to the training bar on the schedule.
  timeline
    .to('.runs', { opacity: faded, ...exit })
    .to('.timing', { opacity: 1, ...enter }, '<0.1')
    .to('.training', { opacity: 1, duration: 0 }, '<')
    .to('.training', { scaleX: 1, ...draw }, '<0.1')
    .to('.planned', { opacity: 1, ...enter }, '>-0.2')
    // Both reasons then return to full strength.
    .to('.column, .runs', { opacity: 1, ...enter }, '>0.3')
    .addLabel('step3')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <defs>
      <marker id="feasibility-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
        <path class="head" d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
    </defs>

    <path class="divider" :d="`M ${divider} 20 V 440`" />

    <g v-for="(reason, index) in reasons" :key="reason.name" :class="`reason-${index}`">
      <circle class="ring" :cx="reason.x + head.radius" :cy="head.y" :r="head.radius" />
      <text class="number" :x="reason.x + head.radius" :y="head.y + 5" text-anchor="middle">{{ index + 1 }}</text>
      <path class="tick" :d="`M ${reason.x + 8} ${head.y} l 5 5 l 9 -10`" />
      <text class="name" :x="reason.x + 40" :y="head.y + 7">{{ reason.name }}</text>
      <text class="detail" :x="reason.x + 40" :y="head.y + 31">{{ reason.detail }}</text>
    </g>

    <!-- Reason 1. Training pilot -->
    <g class="column column-0">
      <path class="chart-axis" :d="`M ${chart.left - 6} 130 V ${chart.base} H ${chart.right + 4}`" />
      <text class="small" :x="(chart.left + chart.right) / 2" :y="chart.base + 22" text-anchor="middle">13 optimizers</text>
      <rect v-for="(bar, index) in bars" :key="index" class="memory-bar" :x="bar.x" :y="chart.base - bar.height" :width="bar.width" :height="bar.height" />
      <path class="limit" :d="`M ${chart.left - 6} ${limit} H ${chart.right + 4}`" />
      <text class="limit-text" :x="chart.left" :y="limit - 8">8 GB</text>
      <g class="pill">
        <rect class="pill-box" :x="chart.left - 6" y="96" width="132" height="26" rx="13" />
        <text class="pill-text" :x="chart.left + 60" y="114" text-anchor="middle">After one change</text>
      </g>
      <g class="reveal fit">
        <text class="fit-text" :x="chart.right - 18" y="115" text-anchor="end">All fit</text>
        <g class="check" :transform="`translate(${chart.right - 4} 109)`">
          <circle r="10" />
          <path d="M -4.5 0 L -1.5 3.8 L 4.5 -3.8" />
        </g>
      </g>
    </g>

    <!-- Reason 2. Fixed workload -->
    <g class="column column-1">
      <g class="runs">
        <text class="count" :x="grid.x" y="105"><tspan class="strong">92</tspan> tuning runs</text>
        <rect v-for="(square, index) in tuning" :key="`tuning-${index}`" class="reveal tuning" :x="square.x" :y="square.y" :width="grid.size" :height="grid.size" rx="2" />
        <text class="count" :x="grid.x" y="230"><tspan class="strong">51</tspan> main runs</text>
        <rect v-for="(square, index) in main" :key="`main-${index}`" class="reveal main" :x="square.x" :y="square.y" :width="grid.size" :height="grid.size" rx="2" />
      </g>
      <g class="reveal timing">
        <path class="link" :d="`M ${grid.x + 105} 304 V 332`" marker-end="url(#feasibility-head)" />
        <text class="bar-name" :x="training.x + training.width + 10" :y="months.axis - 33">Train models</text>
        <path class="month-axis" :d="`M ${months.x} ${months.axis} h ${12 * months.quarter}`" />
        <path v-for="index in 4" :key="index" class="month-axis" :d="`M ${months.x + (index - 1) * 4 * months.quarter} ${months.axis - 4} v 8`" />
        <text v-for="(month, index) in ['Jan', 'Feb', 'Mar']" :key="month" class="small" :x="months.x + (index * 4 + 2) * months.quarter" :y="months.axis + 19" text-anchor="middle">{{ month }}</text>
      </g>
      <rect class="training" :x="training.x" :y="months.axis - 48" :width="training.width" height="20" rx="5" />
      <text class="reveal planned note" :x="grid.x" :y="months.axis + 44">Training time can be planned</text>
    </g>

  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.divider {
  stroke: var(--color-land);
  stroke-width: 1.5;
}

.ring {
  stroke-width: 1.5;
}

.number {
  font-size: 0.95rem;
  font-weight: 700;
}

.tick {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.name {
  font-size: 20px;
  font-weight: 700;
}

.detail {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.head {
  fill: var(--color-subtle);
}

.link {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.small {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.note {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-muted);
}

.count {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.strong {
  font-weight: 700;
  fill: var(--color-ink);
}

.chart-axis {
  fill: none;
  stroke: var(--color-faint);
}

.memory-bar {
  fill: var(--color-ink);
}

.limit {
  stroke: var(--color-ink);
  stroke-width: 1.5;
  stroke-dasharray: 6 4;
}

.limit-text {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-ink);
}

.pill-box {
  fill: var(--color-surface);
  stroke: var(--color-removed);
  stroke-width: 1.5;
}

.pill-text {
  font-size: 0.8rem;
  fill: var(--color-subtle);
}

.fit-text {
  font-size: 0.85rem;
  font-weight: 700;
  fill: var(--color-positive);
}

.check circle {
  fill: var(--color-positive);
}

.check path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.tuning {
  fill: var(--color-testing-tint);
  stroke: var(--color-testing);
  stroke-width: 1.2;
}

.main {
  fill: var(--color-testing);
}

.training {
  fill: var(--color-testing);
}

.bar-name {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-ink);
}

.month-axis {
  fill: none;
  stroke: var(--color-removed);
  stroke-width: 1.5;
}
</style>
