<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, moveLong, pop } from '../shared/motion'

const page = { width: 190, height: 270, top: 60, fold: 24 }
const pages = [
  { name: 'Chapter 1', title: 'Introduction', x: 210, marked: [1, 5] },
  { name: 'Chapter 2', title: 'Literature Review', x: 430, marked: [3] },
  { name: 'Chapter 3', title: 'Methodology', x: 650, marked: [0, 4, 6] },
]
const lineCount = 7
const widths = [150, 132, 158, 120, 146, 128, 140, 110]
const lineWidth = (x: number, line: number) => widths[(line + x / 10) % widths.length]
const lineY = (line: number) => page.top + 80 + line * 22
const tagY = page.top + page.height - 34
const outline = (x: number) => `M ${x} ${page.top} h ${page.width - page.fold} l ${page.fold} ${page.fold} v ${page.height - page.fold} h ${-page.width} z`
const fold = (x: number) => `M ${x + page.width - page.fold} ${page.top} v ${page.fold} h ${page.fold}`

let count = 0
const marks = pages.flatMap(item => item.marked.map(line => ({
  x: item.x,
  y: lineY(line),
  width: lineWidth(item.x, line),
  pin: item.x + page.width + 14,
  number: ++count,
})))

const approval = { x: 80, y: 190, half: 26, rail: 38 }
// The large opening words, and the centre of the small label they shrink into.
// The middle of the words sits about 25 units above their baseline, so nothing is measured on load.
const headline = { x: 442, y: 250, middle: 225 }
const label = { x: approval.x, y: approval.y + 58 }

// The revision row on the schedule at the bottom.
const schedule = { left: 210, width: 630, axis: 424 }
const unit = schedule.width / 24
const landed = { x: schedule.left, y: schedule.axis - 30, width: 2 * unit, height: 20 }
const nameAt = { x: landed.x + landed.width + 12, y: landed.y + 16 }
const months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
const faded = 0.3

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.pages, .approve, .comments, .revise > *, .axis, .row', { opacity: 0 })
  gsap.set('.revise .new', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.approve .mark', { scale: 0.4, transformOrigin: '50% 50%' })

  timeline
    .addLabel('step0')
    // Click 1. The words shrink into the label of the approval mark, and the chapters appear.
    .addLabel('shrink')
    .to('.headline', { x: label.x - headline.x, y: label.y - headline.middle, scale: 0.2, svgOrigin: `${headline.x} ${headline.middle}`, opacity: 0, ...moveLong }, 'shrink')
    .to('.approve', { opacity: 1, ...enter }, `shrink+=${moveLong.duration - 0.3}`)
    .to('.approve .mark', { scale: 1, ...pop }, '<')
    .to('.pages', { opacity: faded, ...enter }, '<0.1')
    .addLabel('step1')
    // Click 2. Pins mark the comments from the panel.
    .to('.approve', { opacity: faded, ...exit })
    .to('.comments', { opacity: 1, ...enter })
    .addLabel('step2')
    // Click 3. The marked lines are rewritten and the pins become checks.
    .to('.comments, .marked, .draft', { opacity: 0, ...exit })
    .to('.revise .old', { opacity: 1, ...enter })
    .to('.revise .new', { opacity: 1, duration: 0 }, '<0.15')
    .to('.revise .new', { scaleX: 1, ...draw, stagger: 0.06 }, '<')
    .to('.revise .check, .revise .revised', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step3')
    // Click 4. Everything returns at full strength, and the revision takes its place on the schedule.
    .to('.pages, .approve', { opacity: 1, ...enter })
    .to('.axis, .row', { opacity: 1, ...enter }, '<0.1')
    .addLabel('step4')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <text class="headline" :x="headline.x" :y="headline.y" text-anchor="middle">Proposal approved</text>

    <g class="pages">
      <g v-for="item in pages" :key="item.x">
        <path class="sheet" :d="outline(item.x)" />
        <path class="sheet" :d="fold(item.x)" />
        <text class="chapter" :x="item.x + 16" :y="page.top + 30">{{ item.name }}</text>
        <text class="title" :x="item.x + 16" :y="page.top + 52">{{ item.title }}</text>
        <rect
          v-for="line in lineCount" :key="line" class="text-line" :class="{ marked: item.marked.includes(line - 1) }"
          :x="item.x + 16" :y="lineY(line - 1)" :width="lineWidth(item.x, line - 1)" height="7" rx="2"
        />
        <g class="draft">
          <rect class="tag" :x="item.x + 16" :y="tagY" width="104" height="22" rx="11" />
          <text class="tag-text" :x="item.x + 68" :y="tagY + 15" text-anchor="middle">Proposal draft</text>
        </g>
      </g>
    </g>

    <g class="approve">
      <path class="mark" :d="`M ${approval.x} ${approval.y - approval.half} l ${approval.half} ${approval.half} l ${-approval.half} ${approval.half} l ${-approval.half} ${-approval.half} z`" />
      <text class="strong" :x="approval.x" :y="approval.y + 54" text-anchor="middle">Proposal</text>
      <text class="strong" :x="approval.x" :y="approval.y + 72" text-anchor="middle">approved</text>
      <path class="rail" :d="`M ${approval.x} ${approval.y - approval.half - 2} V ${approval.rail} H ${pages[2].x + page.width / 2}`" />
      <g v-for="item in pages" :key="item.x">
        <path class="rail" :d="`M ${item.x + page.width / 2} ${approval.rail} V ${page.top - 14}`" />
        <path class="mark" :d="`M ${item.x + page.width / 2} ${page.top - 3} l -6 -11 h 12 z`" />
      </g>
    </g>

    <g class="comments">
      <g v-for="mark in marks" :key="mark.number">
        <rect class="highlight" :x="mark.x + 11" :y="mark.y - 5" :width="mark.width + 10" height="17" rx="3" />
        <path class="connector" :d="`M ${mark.x + 21 + mark.width} ${mark.y + 3} H ${mark.pin - 11}`" />
        <circle class="pin" :cx="mark.pin" :cy="mark.y + 3" r="11" />
        <text class="pin-number" :x="mark.pin" :y="mark.y + 7.5" text-anchor="middle">{{ mark.number }}</text>
      </g>
      <text class="note" :x="pages[0].x" :y="page.top + page.height + 26">Numbered pins mark each comment from the panel</text>
    </g>

    <g class="revise">
      <rect v-for="mark in marks" :key="`old-${mark.number}`" class="old" :x="mark.x + 16" :y="mark.y - 3" :width="mark.width" height="7" rx="2" />
      <rect v-for="mark in marks" :key="`new-${mark.number}`" class="new" :x="mark.x + 16" :y="mark.y + 7" :width="mark.width - 12" height="7" rx="2" />
      <g v-for="mark in marks" :key="`check-${mark.number}`" class="check">
        <circle class="check-ring" :cx="mark.pin" :cy="mark.y + 3" r="11" />
        <path class="check-tick" :d="`M ${mark.pin - 5} ${mark.y + 3} l 3.5 3.5 l 6 -7`" />
      </g>
      <g v-for="item in pages" :key="`revised-${item.x}`" class="revised">
        <rect class="tag done" :x="item.x + 16" :y="tagY" width="72" height="22" rx="11" />
        <text class="tag-text strong-tag" :x="item.x + 52" :y="tagY + 15" text-anchor="middle">Revised</text>
      </g>
    </g>

    <g class="axis">
      <path class="axis-line" :d="`M ${schedule.left} ${schedule.axis} h ${schedule.width}`" />
      <g v-for="(month, index) in months" :key="month">
        <path class="axis-line" :d="`M ${schedule.left + index * 4 * unit} ${schedule.axis - 4} v 8`" />
        <text class="month" :x="schedule.left + (index * 4 + 2) * unit" :y="schedule.axis + 20" text-anchor="middle">{{ month }}</text>
      </g>
    </g>

    <g class="row">
      <rect class="row-bar" :x="landed.x" :y="landed.y" :width="landed.width" :height="landed.height" rx="5" />
      <text class="row-label" :x="nameAt.x" :y="nameAt.y">Revise Chapters 1 to 3, November 1 to 15</text>
    </g>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.headline {
  font-size: var(--text-display);
  font-weight: 700;
  letter-spacing: -0.01em;
  fill: var(--color-ink);
}

.sheet {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 1.6;
}

.chapter {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.title {
  font-size: 17px;
  font-weight: 700;
  fill: var(--color-ink);
}

.text-line {
  fill: var(--color-faint);
}

.tag {
  fill: none;
  stroke: var(--color-subtle);
}

.tag.done {
  stroke: var(--color-ink);
  stroke-width: 1.4;
}

.tag-text {
  font-size: 0.8rem;
  fill: var(--color-muted);
}

.strong-tag {
  font-weight: 700;
  fill: var(--color-ink);
}

.mark {
  fill: var(--color-subtle);
}

.strong {
  font-size: 0.9rem;
  font-weight: 700;
  fill: var(--color-ink);
}

.rail {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.highlight {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.3;
}

.connector {
  stroke: var(--color-subtle);
  stroke-width: 1.2;
}

.pin {
  fill: var(--color-subtle);
}

.pin-number {
  font-size: 0.8rem;
  font-weight: 700;
  fill: var(--color-surface);
}

.note {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.old {
  fill: var(--color-surface);
  stroke: var(--color-faint);
  stroke-dasharray: 4 3;
}

.new {
  fill: var(--color-muted);
}

.check-ring {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.check-tick {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.8;
}

.axis-line {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.month {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.row-bar {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.row-label {
  font-size: var(--text-2xs);
  font-weight: 500;
  fill: var(--color-ink);
}
</style>
