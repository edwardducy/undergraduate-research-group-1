<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, moveShort, moveLong } from '../shared/motion'

const width = 444
const mid = width / 2

interface Pipeline { x: number, name: string, width: number, outputs: string[] }

const singles: Pipeline[] = [
  { x: mid - 80, name: 'Model', width: 116, outputs: ['classification'] },
  { x: mid + 80, name: 'Model', width: 116, outputs: ['ner'] },
]
const singlesTop = 20
const single = { top: 116 + singlesTop, width: 116, arrowRoot: 184 + singlesTop, arrowTip: 278 + singlesTop }
const final: Pipeline[] = [
  { x: 106, name: 'Multi-task model', width: 212, outputs: ['classification', 'ner'] },
  { x: 284, name: 'Model', width: 92, outputs: ['classification'] },
  { x: 398, name: 'Model', width: 92, outputs: ['ner'] },
]

function outputX(outputs: string[], index: number) {
  return outputs.length === 1 ? 0 : (index === 0 ? -34 : 34)
}

const angle = 60
const reach = 170
const box = { top: 124, middle: 156, bottom: 188 }
const tip = { x: reach * Math.sin((angle * Math.PI) / 180), y: box.bottom + reach * Math.cos((angle * Math.PI) / 180) }

const optimizers = ['Equal', 'Tuned static', 'Random', 'Uncertainty', 'FAMO', 'GradNorm', 'MGDA',
  'GradDrop', 'IMTL', 'PCGrad', 'CAGrad', 'Nash-MTL', 'PCGrad with uncertainty']
const cell = width / 3
const copies = optimizers.map((name, index) => {
  const last = index === optimizers.length - 1
  return { name, x: last ? mid : cell * (index % 3 + 0.5), y: 10 + Math.floor(index / 3) * 91, width: last ? 250 : 132 }
})
const copyReach = 40

const backToFront = copies.map((copy, index) => ({ ...copy, index })).reverse()
const copyWidth = copies[0].width

const finalBoxes = final.map(pipeline => ({ x: pipeline.x - pipeline.width / 2, y: 116, width: pipeline.width, height: 64, rx: 12 }))

const parts = (scope: string) => `${scope} text, ${scope} path, ${scope} rect`

const halves = [mid - 80, mid + 80].map(x => x - single.width / 2)
const arms = [-80, 80].map(x => ({ x, y: single.arrowRoot - box.bottom }))
const arrowLength = single.arrowTip - single.arrowRoot

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()
  const ink = { color: colour('ink'), ...enter }

  gsap.set(`.singles, .multi, .copy, .legend, ${parts('.final')}`, { opacity: 0 })
  gsap.set('.item, .item .negative', { color: colour('faint') })

  gsap.set('.whole, .multi-name, .midline, .pull .label, .copy-0 .optimizer', { opacity: 0 })
  halves.forEach((x, index) => gsap.set(`.half-${index}`, { attr: { x, y: single.top, width: single.width } }))
  arms.forEach(({ x, y }, index) => gsap.set(`.arm-${index}`, { x, y, color: colour('ink'), svgOrigin: '0 0' }))
  gsap.set('.multi .shaft', { attr: { y2: arrowLength - 12 }, strokeWidth: 2 })
  gsap.set('.multi .head', { y: arrowLength, scaleX: 14 / 12, scaleY: 12 / 16, transformOrigin: '50% 100%' })

  timeline
    .addLabel('step0')
    .to('.item-0', ink)
    .to('.singles, .legend', { opacity: 1, ...enter }, '<')
    .addLabel('step1')
    .set('.singles .model, .singles .arrow', { opacity: 0 }, '+=0.01')
    .set('.multi', { opacity: 1 }, '<')
    .to('.item-1', ink, '<')
    .to('.singles, .legend', { opacity: 0, ...exit }, '<')
    .to('.half-0', { attr: { x: mid - 128, y: box.top, width: 128 }, ...moveShort })
    .to('.half-1', { attr: { x: mid, y: box.top, width: 128 }, ...moveShort }, '<')
    .to('.arm', { x: 0, y: 0, ...moveShort }, '<')
    .set('.half', { opacity: 0 })
    .set('.whole', { opacity: 1 }, '<')
    .to('.multi-name', { opacity: 1, ...enter }, '<')
    .to('.arm-0', { rotation: angle, color: colour('classification'), ...moveLong }, '<-0.25')
    .to('.arm-1', { rotation: -angle, color: colour('ner'), ...moveLong }, '<')
    .to('.multi .shaft', { attr: { y2: reach - 14 }, strokeWidth: 3, ...moveLong }, '<')
    .to('.multi .head', { y: reach, scaleX: 1, scaleY: 1, ...moveLong }, '<')
    .to('.midline, .pull .label', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step2')
    .to('.item-2', ink)
    .to('.item-2 .negative', { color: colour('negative'), ...enter }, '<')
    .to('.multi-name, .midline, .pull .label', { opacity: 0, ...exit }, '<')
    .to('.whole', { attr: { x: copies[0].x - copies[0].width / 2, y: copies[0].y + 6, width: copies[0].width, height: 36, rx: 8 }, ...moveShort }, '<')
    .to('.arm', { x: copies[0].x - mid, y: copies[0].y + 42 - box.bottom, ...moveShort }, '<')
    .to('.multi .shaft', { attr: { y2: copyReach - 10 }, strokeWidth: 2.5, ...moveShort }, '<')
    .to('.multi .head', { y: copyReach, scaleX: 0.75, scaleY: 0.75, ...moveShort }, '<')
    .set('.multi', { opacity: 0 })
    .set('.copy-0', { opacity: 1 }, '<')
    .to('.copy-0 .optimizer', { opacity: 1, ...enter }, '<')
    .to(copies.slice(1).map((_, index) => `.copy-${index + 1}`), { opacity: 1, ...enter, stagger: 0.04 }, '<')
    .addLabel('step3')
    .to('.item-3', ink)
    .to('.copy .optimizer, .copy .pull', { opacity: 0, ...exit }, '<')
    .to('.copy-box', { attr: { y: copies[0].y + 6 }, ...moveShort })
    .to('.copy-12 .copy-box', { attr: { x: mid - copyWidth / 2, width: copyWidth }, ...moveShort }, '<')
    .set(Array.from({ length: copies.length - 3 }, (_, index) => `.copy-${index + 3}`).join(', '), { opacity: 0 })
    .add(finalBoxes.map((attr, index) => gsap.to(`.copy-${index} .copy-box`, { attr, ...moveShort })))
    .set('.copy', { opacity: 0 })
    .set('.final-box', { opacity: 1 }, '<')
    .to(`${parts('.final')}, .legend`, { opacity: 1, ...enter }, '<')
    .addLabel('step4')
})
</script>

<template>
  <div ref="root" class="specific-objectives">
    <section>
      <h3>Specific objectives</h3>
      <ol>
        <li class="item item-0"><b>1</b><span>Train one single&#8209;task model for each task and measure how well it performs.</span></li>
        <li class="item item-1"><b>2</b><span>Design and implement the multi&#8209;task model.</span></li>
        <li class="item item-2"><b>3</b><span>Select one of 13 optimizers by comparing <strong class="negative">negative transfer</strong> and training cost.</span></li>
        <li class="item item-3"><b>4</b><span>Compare the final model with the single&#8209;task models in accuracy, speed, and memory.</span></li>
      </ol>
    </section>

    <svg :viewBox="`0 0 ${width} 472`" :width="width" height="472">
      <g class="multi">
        <path class="midline" :d="`M ${mid} ${box.top - 56} V ${box.bottom + 210}`" />
        <rect v-for="(_, index) in halves" :key="index" class="model half" :class="`half-${index}`" height="64" rx="12" />
        <rect class="model whole" :x="mid - 128" :y="box.top" width="256" height="64" rx="12" />
        <text class="name multi-name" :x="mid" :y="box.middle + 8" text-anchor="middle">Multi-task model</text>
        <g v-for="(task, side) in ['classification', 'ner']" :key="task" class="pull" :class="task">
          <g :transform="`translate(${mid} ${box.bottom})`">
            <g class="arm" :class="`arm-${side}`">
              <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
              <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
            </g>
          </g>
          <text class="tag label" text-anchor="middle">
            <tspan :x="side === 0 ? mid - tip.x : mid + tip.x" :y="tip.y + 34">{{ side === 0 ? 'Humanitarian' : 'Named entity' }}</tspan>
            <tspan :x="side === 0 ? mid - tip.x : mid + tip.x" :y="tip.y + 56">{{ side === 0 ? 'classification' : 'recognition' }}</tspan>
          </text>
        </g>
      </g>

      <g v-for="(group, row) in [singles, final]" :key="row" :class="row === 0 ? 'singles' : 'final'">
        <g v-for="(pipeline, index) in group" :key="index" :class="`pipeline-${index}`">
          <g :transform="`translate(${pipeline.x} ${row === 0 ? singlesTop : 0})`">
            <text class="post" y="32" text-anchor="middle">Post</text>
            <path d="M 0 46 V 100" />
            <path class="head" d="M 0 112 l -7 -12 h 14 z" />

            <rect class="model" :class="{ 'final-box': row === 1 }" :x="-pipeline.width / 2" y="116" :width="pipeline.width" height="64" rx="12" />
            <text class="name" y="156" text-anchor="middle">{{ pipeline.name }}</text>

            <g v-for="(output, slot) in pipeline.outputs" :key="output" :class="output">
              <path class="arrow" :d="`M 0 184 V 202 H ${outputX(pipeline.outputs, slot)} V 266`" />
              <path class="arrow head" :d="`M ${outputX(pipeline.outputs, slot)} 278 l -7 -12 h 14 z`" />
              <rect class="swatch" :x="outputX(pipeline.outputs, slot) - 14" y="284" width="28" height="28" rx="4" />
            </g>
          </g>
        </g>
      </g>
      <g class="final">
        <text class="tag final-tag" x="106" y="350" text-anchor="middle">Final model</text>
        <text class="tag single-tag" x="341" y="350" text-anchor="middle">Single-task models</text>
      </g>

      <g class="legend">
        <g :transform="`translate(${mid - 172} 0)`">
          <rect class="swatch classification" x="0" y="416" width="14" height="14" rx="3" />
          <text class="tag" x="22" y="429">Humanitarian classification</text>
        </g>
        <g :transform="`translate(${mid - 157} 0)`">
          <rect class="swatch ner" x="0" y="444" width="14" height="14" rx="3" />
          <text class="tag" x="22" y="457">Named entity recognition</text>
        </g>
      </g>

      <g v-for="copy in backToFront" :key="copy.name" class="copy" :class="`copy-${copy.index}`">
        <rect class="model copy-box" :x="copy.x - copy.width / 2" :y="copy.y + 6" :width="copy.width" height="36" rx="8" />
        <text class="optimizer" :x="copy.x" :y="copy.y + 30" text-anchor="middle">{{ copy.name }}</text>
        <g v-for="(task, side) in ['classification', 'ner']" :key="task" class="pull" :class="task">
          <g :transform="`translate(${copy.x} ${copy.y + 42}) rotate(${side === 0 ? angle : -angle})`">
            <line class="shaft copy-shaft" x1="0" y1="0" x2="0" :y2="copyReach - 10" />
            <path class="head" :transform="`translate(0 ${copyReach}) scale(0.75)`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.specific-objectives {
  display: flex;
  align-items: center;
  gap: 56px;
  height: 472px;
}

section {
  flex: none;
  width: 384px;
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
  font-size: var(--text-sm);
  grid-template-columns: 2rem 1fr;
  line-height: var(--leading-list);
}

li b {
  font-weight: 700;
}

.negative {
  font-weight: 600;
}

svg {
  flex: none;
  overflow: visible;
}

.post {
  font-size: var(--text-base);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.optimizer {
  font-size: var(--text-2xs);
  font-weight: 600;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.pull .tag {
  fill: currentColor;
}

path {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head {
  fill: var(--color-ink);
  stroke: none;
}

.pull .head {
  fill: currentColor;
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

.copy-shaft {
  stroke-width: 2.5;
}

.swatch {
  fill: currentColor;
}

.classification {
  color: var(--color-classification);
}

.ner {
  color: var(--color-ner);
}

.midline {
  stroke: var(--color-muted);
  stroke-dasharray: 8 6;
}
</style>
