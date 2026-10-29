<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit } from '../shared/motion'

const source = { x: 0, width: 180, height: 70 }
const gate = { x: 390, halfWidth: 100, halfHeight: 58 }
const join = { x: 600, y: 185 }
const finish = { x: 690, y: 140, width: 190, height: 90 }
const lanes = [
  { key: 'data', tag: 'Data', y: 90, number: 1, question: ['About 3,900', 'posts?'], loop: 'Add public Facebook posts' },
  { key: 'labels', tag: 'Labels', y: 280, number: 2, question: ['Agreement', 'targets met?'], loop: 'Revise guidelines, label a new batch' },
]
const diamond = (y: number) => `M ${gate.x - gate.halfWidth} ${y} L ${gate.x} ${y - gate.halfHeight} L ${gate.x + gate.halfWidth} ${y} L ${gate.x} ${y + gate.halfHeight} Z`
const sourceCentre = source.x + source.width / 2
const loop = (y: number) => {
  const below = y + gate.halfHeight
  const bottom = below + 40
  const top = y + source.height / 2
  return {
    line: `M ${gate.x} ${below + 2} V ${bottom} H ${sourceCentre} V ${top + 14}`,
    head: `M ${sourceCentre} ${top + 3} l -6 11 h 12 z`,
    no: { x: gate.x + 10, y: below + 24 },
    label: { x: (gate.x + sourceCentre) / 2, y: bottom - 8 },
  }
}
const faded = 0.3

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.lane, .loop, .pass', { opacity: 0 })

  timeline
    .addLabel('step0')
    // Click 1. The data lane fills the first gate.
    .to('.empty-1', { opacity: 0, ...exit })
    .to('.lane-data', { opacity: 1, ...enter }, '<')
    .addLabel('step1')
    // Click 2. Too few posts loop back for Facebook posts.
    .to('.lane-data', { opacity: faded, ...exit })
    .to('.loop-data', { opacity: 1, ...enter }, '<')
    .addLabel('step2')
    // Click 3. The labels lane fills the second gate.
    .to('.loop-data', { opacity: faded, ...exit })
    .to('.empty-2', { opacity: 0, ...exit }, '<')
    .to('.lane-labels', { opacity: 1, ...enter }, '<')
    .addLabel('step3')
    // Click 4. Missed targets loop back to the guidelines.
    .to('.lane-labels', { opacity: faded, ...exit })
    .to('.loop-labels', { opacity: 1, ...enter }, '<')
    .addLabel('step4')
    // Click 5. Both gates pass and full labeling starts.
    .to('.lane, .loop', { opacity: 1, ...enter })
    .to('.pass', { opacity: 1, ...enter }, '<0.1')
    .to('.finish-box', { stroke: colour('ink'), strokeWidth: 2.5, strokeDasharray: '12 0', ...enter }, '<')
    .to('.finish-mark', { fill: colour('subtle'), ...enter }, '<')
    .to('.finish-name', { fill: colour('ink'), ...enter }, '<')
    .addLabel('step5')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <g v-for="lane in lanes" :key="`empty-${lane.number}`" :class="`empty-${lane.number}`">
      <path class="empty-gate" :d="diamond(lane.y)" />
      <text class="empty-number" :x="gate.x" :y="lane.y + 7" text-anchor="middle">{{ lane.number }}</text>
    </g>

    <g class="finish">
      <rect class="finish-box" :x="finish.x" :y="finish.y" :width="finish.width" :height="finish.height" rx="12" />
      <path class="finish-mark" :d="`M ${finish.x + 32} ${finish.y + 33} l 10 10 l -10 10 l -10 -10 z`" />
      <text class="finish-name" :x="finish.x + 54" :y="finish.y + 40">Full labeling</text>
      <text class="finish-date" :x="finish.x + 54" :y="finish.y + 64">December</text>
    </g>

    <g v-for="lane in lanes" :key="lane.key" class="lane" :class="`lane-${lane.key}`">
      <text class="lane-tag" :x="source.x" :y="lane.y - source.height / 2 - 12">{{ lane.tag }}</text>
      <rect class="source" :x="source.x" :y="lane.y - source.height / 2" :width="source.width" :height="source.height" rx="10" />
      <template v-if="lane.key === 'data'">
        <text class="source-name" :x="sourceCentre" :y="lane.y - 5" text-anchor="middle">Posts from X</text>
        <text class="source-detail" :x="sourceCentre" :y="lane.y + 16" text-anchor="middle">Selected cyclones</text>
      </template>
      <template v-else>
        <text class="source-name" :x="sourceCentre" :y="lane.y - 5" text-anchor="middle">Practice batch</text>
        <rect class="class-tag" :x="source.x + 56" :y="lane.y + 7" width="16" height="11" rx="2" />
        <rect class="span" :x="source.x + 82" :y="lane.y + 7" width="40" height="11" rx="2" />
      </template>
      <path class="arrow" :d="`M ${source.x + source.width + 4} ${lane.y} H ${gate.x - gate.halfWidth - 14}`" />
      <path class="head" :d="`M ${gate.x - gate.halfWidth - 3} ${lane.y} l -11 -6 v 12 z`" />
      <path class="gate" :d="diamond(lane.y)" />
      <text v-for="(line, row) in lane.question" :key="line" class="question" :x="gate.x" :y="lane.y - 3 + row * 19" text-anchor="middle">{{ line }}</text>
    </g>

    <g v-for="lane in lanes" :key="`loop-${lane.key}`" class="loop" :class="`loop-${lane.key}`">
      <path class="loop-line" :d="loop(lane.y).line" />
      <path class="loop-head" :d="loop(lane.y).head" />
      <text class="answer" :x="loop(lane.y).no.x" :y="loop(lane.y).no.y">no</text>
      <text class="loop-label" :x="loop(lane.y).label.x" :y="loop(lane.y).label.y" text-anchor="middle">{{ lane.loop }}</text>
    </g>

    <g class="pass">
      <path v-for="lane in lanes" :key="`pass-${lane.key}`" class="pass-line" :d="`M ${gate.x + gate.halfWidth + 2} ${lane.y} H ${join.x} V ${join.y}`" />
      <path class="pass-line" :d="`M ${join.x} ${join.y} H ${finish.x - 14}`" />
      <path class="pass-head" :d="`M ${finish.x - 3} ${join.y} l -11 -6 v 12 z`" />
      <circle class="pass-dot" :cx="join.x" :cy="join.y" r="5" />
      <text v-for="lane in lanes" :key="`yes-${lane.key}`" class="yes" :x="(gate.x + gate.halfWidth + join.x) / 2" :y="lane.y - 8" text-anchor="middle">yes</text>
      <text class="both" :x="join.x + 44" :y="join.y - 10" text-anchor="middle">both met</text>
      <g v-for="lane in lanes" :key="`check-${lane.key}`">
        <circle class="check" :cx="gate.x + gate.halfWidth - 18" :cy="lane.y - gate.halfHeight + 8" r="13" />
        <path class="tick" :d="`M ${gate.x + gate.halfWidth - 24} ${lane.y - gate.halfHeight + 8} l 4 5 l 8 -10`" />
      </g>
    </g>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.empty-gate {
  fill: none;
  stroke: var(--color-faint);
  stroke-width: 1.5;
  stroke-dasharray: 6 5;
}

.empty-number {
  font-size: 20px;
  fill: var(--color-faint);
}

.finish-box {
  fill: var(--color-surface);
  stroke: var(--color-removed);
  stroke-width: 1.5;
  stroke-dasharray: 6 5;
}

.finish-mark {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.finish-name {
  font-size: 17px;
  font-weight: 700;
  fill: var(--color-subtle);
}

.finish-date {
  font-size: 0.9rem;
  fill: var(--color-subtle);
}

.lane-tag {
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.lane-data .lane-tag {
  fill: var(--color-data);
}

.lane-labels .lane-tag {
  fill: var(--color-classification);
}

.source {
  stroke-width: 2;
}

.lane-data .source {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
}

.lane-labels .source {
  fill: var(--color-surface);
  stroke: var(--color-muted);
}

.source-name {
  font-size: 16px;
  font-weight: 700;
  fill: var(--color-ink);
}

.source-detail {
  font-size: 0.8rem;
  fill: var(--color-data);
}

.class-tag {
  fill: var(--color-classification);
}

.span {
  fill: none;
  stroke: var(--color-ner);
  stroke-width: 1.6;
}

.arrow {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.head {
  fill: var(--color-subtle);
}

.gate {
  fill: var(--color-surface);
  stroke-width: 2.5;
}

.lane-data .gate {
  stroke: var(--color-data);
}

.lane-labels .gate {
  stroke: var(--color-muted);
}

.question {
  font-size: 15px;
  font-weight: 700;
  fill: var(--color-ink);
}

.loop-line {
  fill: none;
  stroke-width: 1.8;
  stroke-dasharray: 6 4;
}

.loop-data .loop-line {
  stroke: var(--color-data);
}

.loop-data .loop-head,
.loop-data .loop-label {
  fill: var(--color-data);
}

.loop-labels .loop-line {
  stroke: var(--color-muted);
}

.loop-labels .loop-head,
.loop-labels .loop-label {
  fill: var(--color-muted);
}

.loop-label {
  font-size: 0.85rem;
}

.answer {
  font-size: 0.85rem;
  font-style: italic;
  fill: var(--color-subtle);
}

.pass-line {
  fill: none;
  stroke: var(--color-positive);
  stroke-width: 2.5;
}

.pass-head,
.pass-dot,
.check {
  fill: var(--color-positive);
}

.yes,
.both {
  font-size: 0.85rem;
  fill: var(--color-positive);
}

.yes {
  font-style: italic;
}

.tick {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.6;
  stroke-linecap: round;
  stroke-linejoin: round;
}
</style>
