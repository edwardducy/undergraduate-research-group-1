<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { angle, model, reach, tip } from '../shared/model'
import { enter, exit, draw, moveLong } from '../shared/motion'

interface Node { id: string, step: number, x: number, y: number, width: number, height: number, label: string }

const columns = [134, 442, 750]
const [left, middle, right] = columns
const width = 220
const height = 34
const box = (id: string, step: number, x: number, y: number, label: string, size = { width, height }) =>
  ({ id, step, x, y, label, ...size })
const fork = { width: 124, height: 50 }

const gap = 36
const top = 64
const row = (index: number) => top + index * (height + gap)
const saved = row(1) + fork.height + gap
const below = (y: number) => y + height + gap
const selection = below(saved)
const evaluation = below(selection)
const finalTop = below(evaluation)
const bottom = finalTop + height

const nodes: Node[] = [
  box('posts', 2, left, row(0), 'Posts'),
  box('cleaning', 2, left, row(1), 'Cleaning'),
  box('labeling', 2, left, row(2), 'Labeling'),
  box('splitting', 2, left, row(3), 'Splitting'),
  box('dataset', 2, left, row(4), 'Labeled dataset'),

  box('tuning', 3, middle, row(0), 'Fine-tuning'),
  box('single', 3, middle - 68, row(1), 'Single-task\nmodels', fork),
  box('candidates', 3, middle + 68, row(1), 'Candidate\nmodels', fork),
  box('saved', 3, middle, saved, 'Saved models'),
  box('selection', 4, middle, selection, 'Optimizer selection'),
  box('evaluation', 4, middle, evaluation, 'Final evaluation'),

  box('database', 5, right, row(0), 'Database'),
  box('api', 5, right, row(1), 'API server'),
  box('web', 5, right, row(2), 'Web interface'),
  box('sorted', 5, right, row(3), 'Sorted posts\nfor DRRMO staff', { width, height: 50 }),
]

const captions = [
  { step: 1, x: left, lines: ['Data', 'preparation'] },
  { step: 1, x: middle, lines: ['Model training', 'and selection'] },
  { step: 1, x: right, lines: ['Prototype web', 'application'] },
]

const down = (x: number, from: number, to: number) => [[x, from], [x, to]]
const split = top + height + gap / 2
const finalWidth = (256 * height) / 64
interface Arrow { step: number, from: string, to: string, points: number[][], both?: boolean, backward?: boolean }
const arrows: Arrow[] = [
  { step: 2, from: 'posts', to: 'cleaning', points: down(left, row(0) + height, row(1)) },
  { step: 2, from: 'cleaning', to: 'labeling', points: down(left, row(1) + height, row(2)) },
  { step: 2, from: 'labeling', to: 'splitting', points: down(left, row(2) + height, row(3)) },
  { step: 2, from: 'splitting', to: 'dataset', points: down(left, row(3) + height, row(4)) },
  { step: 3, from: 'dataset', to: 'tuning', points: [[left + width / 2, row(4) + height / 2], [(left + middle) / 2, row(4) + height / 2], [(left + middle) / 2, top + height / 2], [middle - width / 2, top + height / 2]] },
  { step: 3, from: 'tuning', to: 'single', points: [[middle, top + height], [middle, split], [middle - 68, split], [middle - 68, row(1)]] },
  { step: 3, from: 'tuning', to: 'candidates', points: [[middle, top + height], [middle, split], [middle + 68, split], [middle + 68, row(1)]] },
  { step: 3, from: 'single', to: 'saved', points: [[middle - 68, row(1) + fork.height], [middle - 68, saved - gap / 2], [middle, saved - gap / 2], [middle, saved]] },
  { step: 3, from: 'candidates', to: 'saved', points: [[middle + 68, row(1) + fork.height], [middle + 68, saved - gap / 2], [middle, saved - gap / 2], [middle, saved]] },
  { step: 4, from: 'saved', to: 'selection', points: down(middle, saved + height, selection) },
  { step: 4, from: 'selection', to: 'evaluation', points: down(middle, selection + height, evaluation) },
  { step: 4, from: 'evaluation', to: 'final', points: down(middle, evaluation + height, finalTop) },
  { step: 5, from: 'final', to: 'api', points: [[middle + finalWidth / 2, finalTop + height / 2], [(middle + right) / 2, finalTop + height / 2], [(middle + right) / 2, row(1) + height / 2], [right - width / 2, row(1) + height / 2]] },
  { step: 5, from: 'api', to: 'database', points: down(right, row(0) + height, row(1)), both: true, backward: true },
  { step: 5, from: 'api', to: 'web', points: down(right, row(1) + height, row(2)), both: true },
  { step: 5, from: 'web', to: 'sorted', points: down(right, row(2) + height, row(3)) },
]

const headLength = 12
function head(from: number[], to: number[]) {
  const angle = (Math.atan2(to[1] - from[1], to[0] - from[0]) * 180) / Math.PI - 90
  return `translate(${to[0]} ${to[1]}) rotate(${angle})`
}
function shorten(from: number[], to: number[]) {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1])
  const k = (length - headLength) / length
  return [from[0] + (to[0] - from[0]) * k, from[1] + (to[1] - from[1]) * k]
}
const speed = 600
const drawn = arrows.map((arrow) => {
  const points = arrow.points.map(point => [...point])
  const length = points.slice(1).reduce((sum, point, index) => sum + Math.hypot(point[0] - points[index][0], point[1] - points[index][1]), 0)
  const ends = [{ from: points[points.length - 2], to: points[points.length - 1] }]
  points[points.length - 1] = shorten(points[points.length - 2], points[points.length - 1])
  if (arrow.both) {
    ends.push({ from: points[1], to: points[0] })
    points[0] = shorten(points[1], points[0])
  }
  return { ...arrow, d: `M ${points.map(point => point.join(' ')).join(' L ')}`, heads: ends.map(end => head(end.from, end.to)), duration: Math.min(Math.max(length / speed, 0.2), 0.6) }
})

const figureScale = 0.92
const drawnCentre = { x: 442, y: bottom / 2 }
const centre = { x: 442, y: 228 }
const figureTransform = `translate(${centre.x} ${centre.y}) scale(${figureScale}) translate(${-drawnCentre.x} ${-drawnCentre.y})`
const small = {
  x: centre.x + (middle - drawnCentre.x) * figureScale,
  y: centre.y + (finalTop + height / 2 - drawnCentre.y) * figureScale,
  scale: (height / 64) * figureScale,
}
const start = tip(reach)
const pulls = [
  { kind: 'classification', rotate: angle, x: model.x - start.x, lines: ['Humanitarian', 'classification'] },
  { kind: 'ner', rotate: -angle, x: model.x + start.x, lines: ['Named entity', 'recognition'] },
]

const post = 'Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!'

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.part, .head, .short-name, .input', { opacity: 0 })
  gsap.set('.line', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.backward .line', { strokeDashoffset: -1 })
  gsap.set('.final', { svgOrigin: `${model.x} ${model.middle}` })

  timeline
    .addLabel('step0')
    .to('.final', { x: small.x - model.x, y: small.y - model.middle, scale: small.scale, ...moveLong })
    .to('.long-name, .pull, .pull-name', { opacity: 0, ...exit }, '<')
    .to('.short-name', { opacity: 1, ...enter }, '<0.3')
    .to('.caption', { opacity: 1, stagger: 0.08, ...enter }, '<0.2')
    .addLabel('step1')

  const lead = 0.15
  for (let step = 2; step <= 5; step++) {
    const start = timeline.duration()
    const shown: Record<string, number> = {}
    for (const node of nodes) {
      if (node.step < step)
        shown[node.id] = -lead
      else if (node.step === step && !drawn.some(arrow => arrow.step === step && arrow.to === node.id))
        shown[node.id] = 0
    }
    shown.final = -lead
    drawn.forEach((arrow, index) => {
      if (arrow.step !== step)
        return
      const begin = start + shown[arrow.from] + lead
      const end = begin + arrow.duration
      timeline
        .to(`.arrow-${index}`, { opacity: 1, duration: 0 }, begin)
        .to(`.arrow-${index} .line`, { strokeDashoffset: 0, duration: arrow.duration, ease: draw.ease }, begin)
      arrow.heads.forEach((_, slot) => {
        const arrives = (slot === 0) !== Boolean(arrow.backward)
        timeline.to(`.arrow-${index} .head-${slot}`, { opacity: 1, duration: 0.1, ease: enter.ease }, arrives ? end - 0.05 : begin)
      })
      if (arrow.to !== 'final')
        shown[arrow.to] = Math.max(shown[arrow.to] ?? 0, end - start - 0.05)
    })
    for (const node of nodes.filter(node => node.step === step))
      timeline.to(`.node-${node.id}`, { opacity: 1, ...enter }, start + shown[node.id])
    timeline.addLabel(`step${step}`)
  }

  timeline
    .to('.part', { opacity: 0, ...exit })
    .to('.final', { x: 0, y: 0, scale: 1, ...moveLong }, '<0.1')
    .to('.short-name', { opacity: 0, ...exit }, '<')
    .to('.long-name', { opacity: 1, ...enter }, '<0.3')
    .to('.input', { opacity: 1, ...enter }, '>-0.1')
    .fromTo('.arrow-in', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.2')
    .addLabel('step6')
    .fromTo('.pull', { opacity: 1, clipPath: 'inset(0% 0% 100% 0%)' }, { opacity: 1, clipPath: 'inset(0% 0% 0% 0%)', immediateRender: false, ...draw }, '+=0.01')
    .to('.pull-name', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step7')
}, 0, true)
</script>

<template>
  <div ref="root" class="system-diagram">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <g :transform="figureTransform">
        <text v-for="caption in captions" :key="caption.x" class="part caption tag" :class="`step-${caption.step}`" text-anchor="middle">
          <tspan v-for="(line, row) in caption.lines" :key="row" :x="caption.x" :y="18 + row * 22">{{ line }}</tspan>
        </text>

        <g v-for="(arrow, index) in drawn" :key="index" class="part arrow" :class="[`step-${arrow.step}`, `arrow-${index}`, { backward: arrow.backward }]">
          <path class="line" :d="arrow.d" pathLength="1" />
          <path v-for="(transform, end) in arrow.heads" :key="end" class="head" :class="`head-${end}`" d="M 0 0 l -7 -12 h 14 z" :transform="transform" />
        </g>

        <g v-for="node in nodes" :key="node.id" class="part node" :class="[`step-${node.step}`, `node-${node.id}`]">
          <rect class="model" :x="node.x - node.width / 2" :y="node.y" :width="node.width" :height="node.height" rx="12" />
          <foreignObject :x="node.x - node.width / 2" :y="node.y" :width="node.width" :height="node.height">
            <div class="label" :class="node.kind">{{ node.label }}</div>
          </foreignObject>
        </g>

      </g>

      <text class="post input" x="81" y="36">{{ post }}</text>
      <g class="input arrow-in">
        <path class="plain" d="M 442 52 V 90" />
        <path class="tip" d="M 442 102 l -7 -12 h 14 z" />
      </g>
      <g class="final">
        <g v-for="pull in pulls" :key="pull.kind" :class="pull.kind">
          <g class="pull" :transform="`translate(${model.x} ${model.bottom}) rotate(${pull.rotate})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <path class="pull-head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag pull-name" text-anchor="middle" :transform="`translate(${pull.x} ${model.bottom + start.y})`">
            <tspan v-for="(line, row) in pull.lines" :key="row" x="0" :y="34 + row * 22">{{ line }}</tspan>
          </text>
        </g>
        <rect class="model" x="314" y="112" width="256" height="64" rx="12" />
        <text class="name long-name" x="442" y="152" text-anchor="middle">Multi-task model</text>
        <text class="name short-name" x="442" y="153" text-anchor="middle">Final model</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: currentColor;
}

.caption {
  font-size: calc(var(--text-2xs) / 0.92);
  color: var(--color-muted);
}

.post {
  font-size: var(--text-base);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.short-name {
  font-size: calc(var(--text-xs) * 64 / 34);
  font-weight: 400;
}

.line,
.plain {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.head,
.tip {
  fill: var(--color-ink);
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.label {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: var(--text-xs);
  line-height: 1.2;
  text-align: center;
  white-space: pre-line;
  color: var(--color-ink);
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

.pull-head {
  fill: currentColor;
}

.ner {
  color: var(--color-ner);
}

.classification {
  color: var(--color-classification);
}
</style>
