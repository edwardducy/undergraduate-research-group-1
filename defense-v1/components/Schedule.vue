<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw } from '../shared/motion'

// One unit is a quarter of a month, from November 2026 to the end of April 2027.
const chart = { left: 310, right: 884, top: 96, row: 42, bar: 22 }
const unit = (chart.right - chart.left) / 24
const x = (time: number) => chart.left + time * unit
const rowY = (index: number) => chart.top + index * chart.row + chart.row / 2
const months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

type Kind = 'coding' | 'testing' | 'data' | 'documentation'
const kinds: { kind: Kind, name: string }[] = [
  { kind: 'coding', name: 'Coding' },
  { kind: 'testing', name: 'Testing' },
  { kind: 'data', name: 'Data' },
  { kind: 'documentation', name: 'Documentation' },
]
let legendX = chart.left
const legend = kinds.map((item) => {
  const at = legendX
  legendX += item.name.length * 9 + 56
  return { ...item, x: at }
})

const tasks: { name: string, step: number, parts: [number, number, Kind][] }[] = [
  { name: 'Revise Chapters 1 to 3', step: 1, parts: [[0, 2, 'documentation']] },
  { name: 'Collect posts, prepare labeling', step: 2, parts: [[2, 4, 'data']] },
  { name: 'Label posts', step: 2, parts: [[4, 11, 'data']] },
  { name: 'Benchmark code and tests', step: 3, parts: [[4, 8, 'coding'], [8, 10, 'testing']] },
  { name: 'Prototype and tests', step: 3, parts: [[5, 9, 'coding'], [9, 11, 'testing']] },
  { name: 'Train models', step: 4, parts: [[11, 15.5, 'testing']] },
  { name: 'Write Chapters 4 and 5', step: 5, parts: [[12, 16, 'documentation']] },
]
const bars = tasks.map((task, index) => {
  const start = task.parts[0][0]
  const end = task.parts[task.parts.length - 1][1]
  return { ...task, index, y: rowY(index) - chart.bar / 2, start: x(start), width: x(end) - x(start), hollow: task.parts[0][2] === 'documentation' }
})
const bottom = chart.top + tasks.length * chart.row
const middle = chart.top + tasks.length * chart.row / 2

const holiday = { start: 7, end: 8.2 }
const defense = { x: x(20.5), y: middle }
const faded = 0.28

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.extra, .bar-wrap, .ending', { opacity: 0 })
  gsap.set('.bar-grow', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.task-name', { fill: colour('faint'), fontWeight: 500 })

  timeline.addLabel('step0')

  // Clicks 1 to 5. New bars draw in at full strength and the bars before them fade.
  const extras: Record<number, string> = { 2: '.extra.holiday', 3: '.extra.alongside', 4: '.extra.needs', 5: '.extra.results' }
  for (let step = 1; step <= 5; step++) {
    if (extras[step - 1])
      timeline.to(extras[step - 1], { opacity: 0, ...exit })
    if (step > 1) {
      timeline
        .to(`.step-${step - 1} .bar-wrap`, { opacity: faded, ...exit }, extras[step - 1] ? '<' : '>')
        .to(`.step-${step - 1} .task-name`, { fill: colour('subtle'), fontWeight: 500, ...exit }, '<')
    }
    timeline
      .to(`.step-${step} .task-name`, { fill: colour('ink'), fontWeight: 700, ...enter })
      .to(`.step-${step} .bar-wrap`, { opacity: 1, duration: 0 }, '<')
      .to(`.step-${step} .bar-grow`, { scaleX: 1, ...draw, stagger: 0.1 }, '<')
    if (extras[step])
      timeline.to(extras[step], { opacity: 1, ...enter }, '>-0.1')
    timeline.addLabel(`step${step}`)
  }

  // Click 6. Every bar returns to full strength, then the end of work, the buffer and the defense appear.
  timeline
    .to(extras[5], { opacity: 0, ...exit })
    .to('.bar-wrap', { opacity: 1, ...enter })
    .to('.task-name', { fill: colour('ink'), fontWeight: 500, ...enter }, '<')
    .to('.ending.finish', { opacity: 1, ...enter })
    .to('.ending.buffer', { opacity: 1, ...enter }, '<0.15')
    .to('.ending.defense', { opacity: 1, ...enter }, '<0.15')
    .addLabel('step6')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <defs>
      <pattern id="schedule-break" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="7" class="break-stripe" />
      </pattern>
      <pattern id="schedule-buffer" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="10" class="buffer-stripe" />
      </pattern>
      <clipPath v-for="bar in bars" :id="`schedule-bar-${bar.index}`" :key="bar.index">
        <rect :x="bar.start" :y="bar.y" :width="bar.width" :height="chart.bar" rx="5" />
      </clipPath>
    </defs>

    <g v-for="item in legend" :key="item.kind" :transform="`translate(${item.x} 6)`">
      <rect class="swatch" :class="item.kind" x="1" y="1" width="16" height="16" rx="3" />
      <text class="legend" x="26" y="15">{{ item.name }}</text>
    </g>

    <text class="year" :x="x(0)" y="60">2026</text>
    <text class="year" :x="x(8)" y="60">2027</text>
    <g v-for="(name, index) in months" :key="name">
      <text class="month" :x="x(index * 4 + 2)" :y="chart.top - 12" text-anchor="middle">{{ name }}</text>
      <path class="grid" :d="`M ${x(index * 4)} ${chart.top - 4} V ${bottom}`" />
    </g>
    <path class="grid" :d="`M ${chart.right - 1} ${chart.top - 4} V ${bottom}`" />

    <g class="ending buffer">
      <rect class="buffer-fill" :x="x(16)" :y="chart.top" :width="x(20) - x(16)" :height="bottom - chart.top" />
      <rect class="buffer-box" :x="x(16.6)" :y="middle - 17" :width="x(19.4) - x(16.6)" height="34" rx="6" />
      <text class="buffer-label" :x="x(18)" :y="middle + 6" text-anchor="middle">Buffer</text>
    </g>

    <g class="extra alongside">
      <rect class="band" :x="x(4)" :y="chart.top + 2 * chart.row + 4" :width="x(11) - x(4)" :height="3 * chart.row - 8" rx="6" />
      <text class="note" :x="x(7.5)" :y="bottom + 24" text-anchor="middle">While labeling runs</text>
    </g>

    <g v-for="bar in bars" :key="bar.name" class="task" :class="`step-${bar.step}`">
      <text class="task-name" :x="chart.left - 16" :y="rowY(bar.index) + 6" text-anchor="end">{{ bar.name }}</text>
      <g class="bar-wrap">
        <g class="bar-grow">
          <rect
            v-if="bar.hollow" class="segment documentation"
            :x="bar.start + 1" :y="bar.y + 1" :width="bar.width - 2" :height="chart.bar - 2" rx="5"
          />
          <g v-else :clip-path="`url(#schedule-bar-${bar.index})`">
            <rect
              v-for="[from, to, kind] in bar.parts" :key="from" class="segment" :class="kind"
              :x="x(from)" :y="bar.y" :width="x(to) - x(from)" :height="chart.bar"
            />
            <line
              v-if="bar.parts.length > 1" class="seam"
              :x1="x(bar.parts[1][0])" :y1="bar.y" :x2="x(bar.parts[1][0])" :y2="bar.y + chart.bar"
            />
            <rect
              v-if="bar.index === 2" class="break"
              :x="x(holiday.start)" :y="bar.y" :width="x(holiday.end) - x(holiday.start)" :height="chart.bar"
            />
          </g>
        </g>
      </g>
    </g>

    <text class="extra holiday note" :x="x((holiday.start + holiday.end) / 2)" :y="bars[2].y - 7" text-anchor="middle">Christmas break</text>

    <g class="extra needs">
      <path class="needs-line" :d="`M ${x(11)} ${bars[2].y + chart.bar + 3} V ${bars[5].y - 10}`" />
      <path class="needs-head" :d="`M ${x(11)} ${bars[5].y - 2} l -6 -10 h 12 z`" />
      <text class="note" :x="x(11) + 12" :y="rowY(4) + 6">Needs the finished dataset</text>
    </g>

    <text class="extra results note" :x="x(14)" :y="bars[6].y + chart.bar + 20" text-anchor="middle">As results come in</text>

    <g class="ending finish">
      <path class="finish-line" :d="`M ${x(16)} ${chart.top - 4} V ${bottom + 8}`" />
      <text class="note" :x="x(16)" :y="bottom + 28" text-anchor="middle">All tasks end</text>
    </g>

    <g class="ending defense">
      <path class="defense-mark" :d="`M ${defense.x} ${defense.y - 14} l 14 14 l -14 14 l -14 -14 z`" />
      <text class="defense-label" :x="defense.x" :y="defense.y + 38" text-anchor="middle">Final</text>
      <text class="defense-label" :x="defense.x" :y="defense.y + 58" text-anchor="middle">defense</text>
    </g>
  </svg>
</template>

<style scoped>
.legend {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.year {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-subtle);
}

.month {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.grid {
  stroke: var(--color-land);
  stroke-width: 1.5;
}

.task-name {
  font-size: var(--text-2xs);
}

.note {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-muted);
}

.swatch.coding,
.segment.coding {
  fill: var(--color-coding);
}

.swatch.testing,
.segment.testing {
  fill: var(--color-testing);
}

.swatch.data,
.segment.data {
  fill: var(--color-data);
}

.swatch.documentation,
.segment.documentation {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.seam {
  stroke: var(--color-surface);
  stroke-width: 2;
}

.break {
  fill: url(#schedule-break);
  opacity: 0.8;
}

.break-stripe {
  stroke: var(--color-surface);
  stroke-width: 2.6;
}

.band {
  fill: var(--color-land);
  opacity: 0.7;
}

.needs-line {
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.needs-head {
  fill: var(--color-subtle);
}

.buffer-fill {
  fill: url(#schedule-buffer);
  opacity: 0.8;
}

.buffer-stripe {
  stroke: var(--color-faint);
  stroke-width: 3;
}

.buffer-box {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.buffer-label {
  font-size: var(--text-2xs);
  font-weight: 700;
  fill: var(--color-muted);
}

.finish-line {
  stroke: var(--color-subtle);
  stroke-width: 2;
  stroke-dasharray: 6 4;
}

.defense-mark {
  fill: var(--color-subtle);
}

.defense-label {
  font-size: 0.85rem;
  font-weight: 700;
  fill: var(--color-muted);
}
</style>
