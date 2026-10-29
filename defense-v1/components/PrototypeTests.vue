<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { draw, enter, pop } from '../shared/motion'

// One test case, read from left to right.
const card = { y: 200, width: 260, height: 132 }
const columns = [
  { name: 'Input', x: 0 },
  { name: 'Expected result', x: 312 },
  { name: 'Actual result', x: 624 },
]
const middle = card.y + card.height / 2
const gap = (index: number) => columns[index].x + card.width + (columns[index + 1].x - columns[index].x - card.width) / 2

// The labels in the prototype's colours for these two categories.
const oldLabel = { name: 'Caution and advice', y: card.y + 20, width: 170 }
const newLabel = { name: 'Requests or urgent needs', width: 214 }
const changedY = card.y + 80
const resultY = card.y + 34
const savedY = card.y + 98

const pass = { width: 112, height: 38 }
const passX = columns[2].x + (card.width - pass.width) / 2
const passY = card.y + card.height + 30

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.expected, .actual, .match, .pass', { opacity: 0 })
  gsap.set('.match, .pass', { scale: 0.6, transformOrigin: '50% 50%' })
  gsap.set('.arrow', { strokeDasharray: 1, strokeDashoffset: 1 })

  timeline
    .addLabel('step0')
    // Click 1. The arrow leads to the result the test case expects.
    .to('.arrow', { strokeDashoffset: 0, ...draw })
    .to('.expected', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step1')
    // Click 2. The actual result matches the expected one, so the test case passes.
    .to('.actual', { opacity: 1, ...enter })
    .to('.match', { opacity: 1, scale: 1, ...pop }, '>-0.05')
    .to('.pass', { opacity: 1, scale: 1, ...pop }, '>0.1')
    .addLabel('step2')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <text class="tag" x="0" y="84">Use case</text>
    <text class="title" x="0" y="118">Correct a label</text>

    <g class="input">
      <text class="tag" :x="columns[0].x" :y="card.y - 16">{{ columns[0].name }}</text>
      <rect class="card" :x="columns[0].x" :y="card.y" :width="card.width" :height="card.height" rx="10" />
      <g class="label old">
        <rect :x="columns[0].x + 18" :y="oldLabel.y" :width="oldLabel.width" height="28" rx="14" />
        <circle :cx="columns[0].x + 34" :cy="oldLabel.y + 14" r="5" />
        <text :x="columns[0].x + 46" :y="oldLabel.y + 19">{{ oldLabel.name }}</text>
        <path class="strike" :d="`M ${columns[0].x + 26} ${oldLabel.y + 14} h ${oldLabel.width - 16}`" />
      </g>
      <path class="change" :d="`M ${columns[0].x + 34} ${oldLabel.y + 33} V ${changedY - 5} M ${columns[0].x + 29} ${changedY - 11} l 5 6 l 5 -6`" />
      <g class="label new">
        <rect :x="columns[0].x + 18" :y="changedY" :width="newLabel.width" height="28" rx="14" />
        <circle :cx="columns[0].x + 34" :cy="changedY + 14" r="5" />
        <text :x="columns[0].x + 46" :y="changedY + 19">{{ newLabel.name }}</text>
      </g>
    </g>

    <path class="arrow" :d="`M ${gap(0) - 18} ${middle} H ${gap(0) + 14}`" pathLength="1" />
    <path class="arrow-head expected" :d="`M ${gap(0) + 8} ${middle - 7} l 8 7 l -8 7`" />

    <g v-for="(column, index) in columns.slice(1)" :key="column.name" :class="index === 0 ? 'expected' : 'actual'">
      <text class="tag" :x="column.x" :y="card.y - 16">{{ column.name }}</text>
      <rect class="card" :x="column.x" :y="card.y" :width="card.width" :height="card.height" rx="10" />
      <g class="label new">
        <rect :x="column.x + 18" :y="resultY" :width="newLabel.width" height="28" rx="14" />
        <circle :cx="column.x + 34" :cy="resultY + 14" r="5" />
        <text :x="column.x + 46" :y="resultY + 19">{{ newLabel.name }}</text>
      </g>
      <g class="saved">
        <circle :cx="column.x + 30" :cy="savedY - 5" r="9" />
        <path :d="`M ${column.x + 26} ${savedY - 5} l 3 3.5 l 5.5 -6.5`" />
        <text :x="column.x + 46" :y="savedY">Saved</text>
      </g>
    </g>

    <g class="match">
      <circle :cx="gap(1)" :cy="middle" r="17" />
      <path :d="`M ${gap(1) - 7} ${middle - 4} h 14 M ${gap(1) - 7} ${middle + 4} h 14`" />
    </g>

    <g class="pass">
      <rect :x="passX" :y="passY" :width="pass.width" :height="pass.height" rx="19" />
      <path :d="`M ${passX + 22} ${passY + 19} l 5 5 l 9 -10`" />
      <text :x="passX + 46" :y="passY + 26">Pass</text>
    </g>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.tag {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.title {
  font-size: var(--text-xl);
  font-weight: 700;
  fill: var(--color-ink);
}

.card {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.label rect {
  fill: var(--color-surface);
  stroke: var(--color-faint);
  stroke-width: 1.5;
}

.label text {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-ink);
}

/* The two label colours match the prototype on the earlier slide. */
.label.old circle {
  fill: #6250d6;
}

.label.old text {
  fill: var(--color-subtle);
}

.label.new circle {
  fill: #2a78d6;
}

.label.new rect {
  stroke: var(--color-ink);
}

.strike {
  stroke: var(--color-negative);
  stroke-width: 2;
}

.change,
.arrow,
.arrow-head {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.change {
  stroke-width: 1.5;
}

.saved circle,
.saved path {
  fill: none;
  stroke: var(--color-positive);
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.saved text {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-positive);
}

.match circle,
.pass rect {
  fill: var(--color-positive);
}

.match path,
.pass path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.pass text {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-surface);
}
</style>
