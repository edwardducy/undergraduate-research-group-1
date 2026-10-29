<script setup lang="ts">
import gsap from 'gsap'
import { MotionPathPlugin } from 'gsap/MotionPathPlugin'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { archive, cells, hinge, lock, post } from '../shared/archive'
import { draw, enter, exit, moveLong } from '../shared/motion'

// Posts start large on the left and shrink as they enter the box.
const startScale = 1.4
const sources = [
  [20, 30], [170, 10], [320, 50], [60, 130], [220, 115], [370, 150],
  [10, 230], [160, 215], [310, 250], [70, 330], [230, 320], [380, 345],
]
const flights = sources.map(([x, y], index) => {
  const cell = cells[index]
  const from = { x: x + (post.width * startScale) / 2, y: y + (post.height * startScale) / 2 }
  const to = { x: cell.x + post.width / 2, y: cell.y + post.height / 2 }
  const bend = index % 2 ? 30 : -30
  const control = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 + bend }
  // The card moves by its top-left corner, so its path is the centre path shifted by half its size.
  const half = (scale: number) => ({ x: (post.width * scale) / 2, y: (post.height * scale) / 2 })
  const offset = (point: { x: number, y: number }, scale: number) => ({ x: point.x - half(scale).x - cell.x, y: point.y - half(scale).y - cell.y })
  const start = offset(from, startScale)
  const bendAt = offset(control, (startScale + 1) / 2)
  const path = [
    start,
    { x: start.x + (2 / 3) * (bendAt.x - start.x), y: start.y + (2 / 3) * (bendAt.y - start.y) },
    { x: (2 / 3) * bendAt.x, y: (2 / 3) * bendAt.y },
    { x: 0, y: 0 },
  ]
  return { start, path, d: `M ${from.x} ${from.y} Q ${control.x} ${control.y} ${to.x} ${to.y}` }
})

const note = { x: archive.x + archive.width / 2, y: archive.y + archive.height + 30 }

// Click 3. The box moves from the right to the centre for the pilot search.
const shift = -250
const centreX = archive.x + shift

// Click 4. Placeholder split until the pilot gives real counts.
const removed = [1, 4, 6, 9, 11]
const targets = [[660, 100, 0.8], [720, 160, 0.6], [775, 220, 0.45], [680, 270, 0.55], [760, 320, 0.35]]
const drifts = removed.map((index, order) => {
  const cell = { x: cells[index].x + shift, y: cells[index].y }
  const [x, y, opacity] = targets[order]
  const start = { x: cell.x + post.width, y: cell.y + post.height / 2 }
  const end = { x: x - 4, y: y + post.height / 2 }
  return {
    index,
    cell,
    x,
    y,
    opacity,
    d: `M ${start.x} ${start.y} Q ${(start.x + end.x) / 2} ${Math.min(cell.y, y) - 20} ${end.x} ${end.y}`,
  }
})

const bars = { x: centreX, unit: 20, collected: 12, kept: 7 }
const faded = 0.3

const root = ref<SVGSVGElement>()

gsap.registerPlugin(MotionPathPlugin)

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.archive-outline', { stroke: colour('subtle'), strokeWidth: 1.8, strokeDasharray: '8 6' })
  gsap.set('.lock-shackle', { stroke: colour('subtle') })
  gsap.set('.lock-body', { fill: colour('subtle') })
  gsap.set('.secured, .flight, .source-label, .card-motion', { opacity: 0 })
  gsap.set('.ghost, .moved, .tag, .drift, .drift-path, .removed-label, .bars', { opacity: 0 })
  gsap.set('.bar', { scaleX: 0, transformOrigin: '0% 50%' })

  const cards = [...root.value!.querySelectorAll<SVGGElement>('.card-motion')]
  cards.forEach((card, index) => gsap.set(card, { ...flights[index].start, scale: startScale, transformOrigin: '0px 0px' }))

  timeline.addLabel('step0')

  // Click 1. Access is secured. The outline turns solid teal and the lock springs open.
  timeline
    .to('.archive-outline', { stroke: colour('data'), strokeWidth: 2.4, strokeDasharray: '14 0', duration: 0.5, ease: 'power2.out' })
    .to('.lock-body', { fill: colour('data'), ...enter }, '<')
    .to('.lock-shackle', { stroke: colour('data'), ...enter }, '<')
    .to('.lock-shackle', { rotation: lock.open, svgOrigin: `${hinge.x} ${hinge.y}`, duration: 0.45, ease: 'back.out(3)' }, '<0.1')
    .to('.locked-note', { opacity: 0, ...exit }, '<')
    .to('.secured', { opacity: 1, ...enter }, '>')
    .addLabel('step1')

  // Click 2. Posts are drawn into the box along their paths and form a pile.
  timeline
    .to('.card-motion', { opacity: 1, ...enter, stagger: 0.04 })
    .to('.flight', { opacity: 0.7, ...enter }, '<0.1')
  const start = timeline.duration()
  // Each post follows its curve and shrinks to its place in the pile.
  cards.forEach((card, index) => {
    timeline.to(card, { motionPath: { path: flights[index].path, type: 'cubic' }, scale: 1, duration: 0.9, ease: 'power2.inOut' }, start + index * 0.07)
  })
  timeline
    .to('.flight', { opacity: 0, ...exit }, '>-0.2')
    .to('.source-label', { opacity: 1, ...enter }, '<')
    .addLabel('step2')

  // Click 3. The box glides to the centre and the pilot search is named.
  timeline
    .to('.source-label, .secured', { opacity: 0, ...exit })
    .to('.ghost, .moved', { opacity: 1, ...enter }, '<')
    .to('.archive', { x: shift, ...moveLong }, '<0.1')
    .to('.tag', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step3')

  // Click 4. Posts removed in cleaning turn grey and drift out. Two bars compare the counts.
    .to('.ghost, .moved', { opacity: 0, ...exit })
    .to('.tag', { opacity: faded, ...exit }, '<')
  removed.forEach((index) => {
    timeline
      .to(`.cell-${index} .card-motion`, { opacity: 0, ...exit }, '<')
      .to(`.cell-${index} .empty-slot`, { opacity: 1, ...enter }, '<')
  })
  timeline
    .to('.drift-path', { opacity: 1, ...enter }, '>')
  drifts.forEach((drift, order) => {
    timeline.fromTo(`.drift-${order}`, { x: drift.cell.x - drift.x, y: drift.cell.y - drift.y, opacity: 1 }, { x: 0, y: 0, opacity: drift.opacity, duration: 0.7, ease: 'power2.out', immediateRender: false }, `<${order ? 0.08 : 0}`)
  })
  timeline
    .to('.removed-label', { opacity: 1, ...enter }, '<')
    .to('.bars', { opacity: 1, duration: 0 }, '<')
    .to('.bar', { scaleX: 1, ...draw, stagger: 0.15 }, '<')
    .addLabel('step4')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <rect class="ghost" :x="archive.x" :y="archive.y" :width="archive.width" :height="archive.height" rx="14" />
    <g class="moved">
      <path class="move-path" :d="`M ${archive.x + archive.width / 2} ${archive.y - 12} Q ${centreX + archive.width / 2 + 125} ${archive.y - 48} ${centreX + archive.width / 2 + 12} ${archive.y - 14}`" />
      <path class="move-head" :d="`M ${centreX + archive.width / 2} ${archive.y - 11} l 13 -8 v 15 z`" />
    </g>

    <path v-for="(flight, index) in flights" :key="index" class="flight" :d="flight.d" />
    <path v-for="drift in drifts" :key="`path-${drift.index}`" class="drift-path" :d="drift.d" />
    <ArchiveBox locked />
    <text class="source-label" :x="archive.x + archive.width / 2" :y="archive.y + archive.height - 14" text-anchor="middle">Most posts come from here</text>
    <text class="note locked-note" :x="note.x" :y="note.y" text-anchor="middle">No access yet</text>
    <text class="note secured" :x="note.x" :y="note.y" text-anchor="middle">Access secured</text>

    <g class="tag">
      <circle class="lens" cx="72" cy="155" r="22" />
      <path class="handle" d="M 88 171 L 106 189" />
      <text class="tag-name" x="22" y="228">Pilot search</text>
      <text class="tag-detail" x="22" y="252">Next 2 to 4 weeks</text>
    </g>

    <path v-for="drift in drifts" :key="`head-${drift.index}`" class="drift-path drift-head" :d="`M ${drift.x - 2} ${drift.y + post.height / 2} l -10 -5 v 10 z`" />
    <rect v-for="(drift, order) in drifts" :key="drift.index" class="drift" :class="`drift-${order}`" :x="drift.x" :y="drift.y" :width="post.width" :height="post.height" rx="6" />
    <text class="removed-label" x="730" y="80" text-anchor="middle">Removed in cleaning</text>

    <g class="bars">
      <text class="bar-name" :x="bars.x - 10" y="384" text-anchor="end">Collected</text>
      <rect class="bar collected" :x="bars.x" y="372" :width="bars.collected * bars.unit" height="15" rx="3" />
      <text class="bar-name" :x="bars.x - 10" y="412" text-anchor="end">After cleaning</text>
      <rect class="bar kept" :x="bars.x" y="400" :width="bars.kept * bars.unit" height="15" rx="3" />
      <path class="bar-mark" :d="`M ${bars.x + bars.kept * bars.unit} 395 V 422`" />
    </g>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.flight {
  fill: none;
  stroke: var(--color-data);
  stroke-width: 1.6;
  stroke-dasharray: 5 5;
}

.source-label {
  font-size: 0.9rem;
  font-weight: 700;
  fill: var(--color-data);
}

.note {
  font-size: 0.95rem;
  font-weight: 700;
}

.locked-note {
  fill: var(--color-subtle);
}

.secured {
  fill: var(--color-data);
}
.ghost {
  fill: none;
  stroke: var(--color-faint);
  stroke-width: 1.5;
  stroke-dasharray: 6 5;
}

.move-path {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.move-head {
  fill: var(--color-subtle);
}

.lens {
  fill: none;
  stroke: var(--color-muted);
  stroke-width: 3;
}

.handle {
  stroke: var(--color-muted);
  stroke-width: 4;
  stroke-linecap: round;
}

.tag-name {
  font-size: 20px;
  font-weight: 700;
  fill: var(--color-ink);
}

.tag-detail {
  font-size: 0.9rem;
  fill: var(--color-subtle);
}

.drift-path {
  fill: none;
  stroke: var(--color-removed);
  stroke-width: 1.3;
  stroke-dasharray: 4 4;
}

.drift-head {
  fill: var(--color-removed);
  stroke: none;
}

.drift {
  fill: none;
  stroke: var(--color-removed);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}

.removed-label {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.bar-name {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.bar.collected {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
  stroke-width: 1.5;
}

.bar.kept {
  fill: var(--color-data);
}

.bar-mark {
  stroke: var(--color-subtle);
  stroke-dasharray: 3 3;
}
</style>
