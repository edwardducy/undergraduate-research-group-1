<script setup lang="ts">
import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { ref, watch } from 'vue'
import { stepCount, useClickTimeline } from '../composables/useClickTimeline'
import { benchmarkChart, benchmarkColumns, benchmarkReach as shortReach, benchmarkX } from '../shared/benchmark'
import { angle, model, modelOrigin, pivot, reach, tilt, tip } from '../shared/model'
import { enter, exit, draw, moveShort, moveLong } from '../shared/motion'

const chart = { ...benchmarkChart, base: 352 }
const chartHeight = chart.base - chart.top
const defaultHeight = benchmarkChart.base - benchmarkChart.top
const columns = benchmarkColumns.map(column => ({
  ...column,
  loss: (column.loss / defaultHeight) * chartHeight,
  cost: (column.cost / defaultHeight) * chartHeight,
}))

const post = 'Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!'

const strongerReach = 320

const start = tip(reach)
const stronger = tip(strongerReach)

const lean = tip(36)
const shift = benchmarkX - model.x

const svg = ref<SVGSVGElement>()
const { $clicks } = useSlideContext()

let decline: gsap.core.Tween | undefined
let declineFrom = 0

function loopDecline(step: number) {
  if (step >= declineFrom)
    decline?.play()
  else
    decline?.pause(0)
}

useClickTimeline(svg, (timeline) => {
  gsap.set('text:not(.example), .swatch, .marker, .midline, .tick, .model-ghost', { opacity: 0 })
  decline = gsap.fromTo('.marker', { y: -3 }, { y: 3, duration: 0.6, ease: 'sine.inOut', repeat: -1, yoyo: true, paused: true })
  gsap.set('.model, .axis', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.result', { y: -28 })
  gsap.set('.result > .classification', { y: -54 })
  gsap.set('.key-classification', { x: 192, y: -64 })

  timeline
    .addLabel('step0')
    .to('.result', { y: -74, ...moveShort })
    .fromTo('.classification .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<')
    .to('.classification .tag, .key-classification > *', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step1')
    .to('.result', { y: -110, ...moveShort })
    .to('.result > .classification', { y: 0, ...moveShort }, '<')
    .to('.key-classification', { x: 0, y: 0, ...moveShort }, '<')
    .fromTo('.ner .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.3')
    .to('.ner .tag, .key-ner > *', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step2')
    .to('.result', { y: 0, ...moveLong })
    .to('.input', { opacity: 1, ...enter }, '<')
    .fromTo('.arrow-in', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.3')
    .to('.model', { strokeDashoffset: 0, ...draw }, '>-0.2')
    .to('.name', { opacity: 1, ...enter }, '<0.2')
    .fromTo('.arrow-out', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '>-0.1')
    .addLabel('step3')
    .to('.input, .arrow, .result', { opacity: 0, ...exit })
    .fromTo('.pull', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.15')
    .to('.pull-classification .tag, .pull-ner .tag', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step4')
    .to('.midline', { opacity: 1, ...enter })
    .to('.pull-ner .shaft', { attr: { y2: strongerReach - 14 }, ...moveShort }, '<')
    .to('.pull-ner .head', { y: strongerReach, ...moveShort }, '<')
    .to('.pull-ner .tag', { x: model.x + stronger.x, y: model.bottom + stronger.y, ...moveShort }, '<')
    .to('.rig', { x: lean.x, y: lean.y, ...moveShort }, '<')
    .to('.model-box', { rotation: tilt, svgOrigin: modelOrigin, ...moveShort }, '<')
    .to('.pulls', { x: pivot.x, y: pivot.y, ...moveShort }, '<')
    .to('.model-ghost', { opacity: 1, ...enter }, '<')
    .addLabel('step5')
    .to('.decline > *', { opacity: 1, ...enter })
    .addLabel('step6')
  declineFrom = stepCount(timeline) - 1
  timeline
    .to('.decline > *', { opacity: 0, ...exit })
    .to('.model-ghost', { opacity: 0, ...exit }, '<')
    .to('.pull-ner .shaft', { attr: { y2: reach - 14 }, ...moveShort }, '<')
    .to('.pull-ner .head', { y: reach, ...moveShort }, '<')
    .to('.pull-ner .tag', { x: model.x + start.x, y: model.bottom + start.y, ...moveShort }, '<')
    .to('.rig', { x: 0, y: 0, ...moveShort }, '<')
    .to('.model-box', { rotation: 0, svgOrigin: modelOrigin, ...moveShort }, '<')
    .to('.pulls', { x: 0, y: 0, ...moveShort }, '<')
    .to('.optimizer, .optimizer > *', { opacity: 1, ...enter }, '<0.2')
    .to('.tick', { opacity: 1, ...enter })
    .addLabel('step7')
    .to('.pull-classification .tag, .pull-ner .tag', { opacity: 0, ...exit })
    .to('.rig, .midline', { x: shift, ...moveLong }, '<')
    .to('.shaft', { attr: { y2: shortReach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: shortReach, ...moveLong }, '<')
    .to('.tick', { y: (shortReach - reach) / 2, ...moveLong }, '<')
    .to('.optimizer', { x: chart.left + chart.width / 2 - model.x, ...moveLong }, '<')
    .to('.axis', { strokeDashoffset: 0, ...draw }, '<0.5')
    .to('.chart text, .chart .swatch', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step8')
    .to('.bar.loss', { scaleY: 1, ...draw, stagger: 0.03 })
    .addLabel('step9')
    .to('.bar.cost', { scaleY: 1, ...draw, stagger: 0.03 })
    .addLabel('step10')
    .to('.column:not(.kept)', { opacity: 0.2, ...exit })
    .to('.kept-name', { opacity: 1, ...enter }, '<')
    .addLabel('step11')

  loopDecline($clicks.value)
})

watch($clicks, loopDecline)
</script>

<template>
  <svg ref="svg" viewBox="0 0 884 456" width="884" height="456">
    <text class="post input" x="81" y="36">{{ post }}</text>

    <g class="arrow arrow-in">
      <path d="M 442 52 V 90" />
      <path class="head" d="M 442 102 l -7 -12 h 14 z" />
    </g>

    <path class="midline" d="M 442 56 V 376" />
    <g class="decline">
      <path class="marker" d="M 311 395 h 18 l -9 16 z" />
      <text class="tag" x="442" y="410" text-anchor="middle">Negative transfer</text>
      <text class="desc" x="442" y="436" text-anchor="middle">Learning both tasks degrades performance</text>
    </g>
    <g class="optimizer">
      <text class="tag" x="442" y="410" text-anchor="middle">Multi-task optimizer</text>
      <text class="desc" x="442" y="436" text-anchor="middle">Balances learning between both tasks</text>
    </g>
    <text class="tag kept-name" x="150" y="410" text-anchor="middle">Optimizer 1</text>

    <rect class="model-ghost" x="314" y="112" width="256" height="64" rx="12" pathLength="624" />
    <g class="rig">
      <g class="model-box">
        <rect class="model" x="314" y="112" width="256" height="64" rx="12" pathLength="1" />
        <text class="name" x="442" y="152" text-anchor="middle">Multi-task model</text>
      </g>

      <g class="pulls">
        <g class="pull-classification">
          <g class="pull" :transform="`translate(442 176) rotate(${angle})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <line class="tick" x1="-9" :y1="reach / 2 - 5" x2="9" :y2="reach / 2 - 5" />
            <line class="tick" x1="-9" :y1="reach / 2 + 5" x2="9" :y2="reach / 2 + 5" />
            <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag" text-anchor="middle" :transform="`translate(${model.x - start.x} ${model.bottom + start.y})`">
            <tspan x="0" y="34">Humanitarian</tspan>
            <tspan x="0" y="56">classification</tspan>
          </text>
        </g>

        <g class="pull-ner">
          <g class="pull" :transform="`translate(442 176) rotate(${-angle})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <line class="tick" x1="-9" :y1="reach / 2 - 5" x2="9" :y2="reach / 2 - 5" />
            <line class="tick" x1="-9" :y1="reach / 2 + 5" x2="9" :y2="reach / 2 + 5" />
            <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag" text-anchor="middle" :transform="`translate(${model.x + start.x} ${model.bottom + start.y})`">
            <tspan x="0" y="34">Named entity</tspan>
            <tspan x="0" y="56">recognition</tspan>
          </text>
        </g>
      </g>
    </g>

    <g class="arrow arrow-out">
      <path d="M 442 186 V 224" />
      <path class="head" d="M 442 236 l -7 -12 h 14 z" />
    </g>

    <g class="result">
      <text class="post example" x="81" y="264">{{ post }}</text>

      <g class="ner">
        <path class="bracket" d="M 253 273 V 279 H 345 V 273" />
        <path class="bracket" d="M 357 273 V 279 H 506 V 273" />
        <text class="tag" x="299" y="305" text-anchor="middle">Location</text>
        <text class="tag" x="431.5" y="305" text-anchor="middle">Request</text>
      </g>

      <g class="classification">
        <path class="bracket" d="M 81 327 V 333 H 803 V 327" />
        <text class="tag" x="442" y="361" text-anchor="middle">Requests or urgent needs</text>
      </g>

      <g class="legend">
        <g class="key-classification">
          <rect class="swatch classification" x="77" y="403" width="14" height="14" rx="3" />
          <text class="tag" x="101" y="404">Humanitarian classification</text>
          <text class="desc" x="101" y="426">Assigns each post one category</text>
        </g>
        <g class="key-ner">
          <rect class="swatch ner" x="476" y="403" width="14" height="14" rx="3" />
          <text class="tag" x="500" y="404">Named entity recognition</text>
          <text class="desc" x="500" y="426">Picks out key details inside the post</text>
        </g>
      </g>
    </g>
    <g class="chart">
      <rect class="swatch loss" :x="chart.left" :y="chart.top - 50" width="14" height="14" rx="3" />
      <text class="tag" :x="chart.left + 22" :y="chart.top - 37">Negative transfer</text>
      <rect class="swatch cost" :x="chart.left" :y="chart.top - 26" width="14" height="14" rx="3" />
      <text class="tag" :x="chart.left + 22" :y="chart.top - 13">Training time and memory</text>

      <path class="axis" :d="`M ${chart.left} ${chart.top} V ${chart.base} H ${chart.left + chart.width}`" pathLength="1" />
      <text class="tag" text-anchor="middle" :transform="`translate(${chart.left - 16} ${(chart.top + chart.base) / 2}) rotate(-90)`">Lower is better</text>

      <g v-for="(column, index) in columns" :key="index" class="column" :class="{ kept: index === 0 }">
        <rect class="bar loss" :x="column.x - 15" :y="chart.base - column.loss" width="14" :height="column.loss" />
        <rect class="bar cost" :x="column.x + 1" :y="chart.base - column.cost" width="14" :height="column.cost" />
        <text class="number" :x="column.x" :y="chart.base + 24" text-anchor="middle">{{ index + 1 }}</text>
      </g>
    </g>
  </svg>
</template>

<style scoped>
.post {
  font-size: var(--text-base);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: currentColor;
}

.desc {
  font-size: var(--text-2xs);
  fill: var(--color-subtle);
}

.model,
path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
}

.model-ghost {
  fill: none;
  stroke: var(--color-negative);
  stroke-width: 2;
  stroke-dasharray: 6 6;
}

.head,
.marker {
  fill: currentColor;
  stroke: none;
}

.bracket {
  stroke-width: 2.5;
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

.tick {
  stroke: currentColor;
  stroke-width: 2.5;
}

.ner,
.pull-ner {
  color: var(--color-ner);
}

.classification,
.pull-classification {
  color: var(--color-classification);
}

.model {
  fill: var(--color-surface);
}

.midline {
  color: var(--color-muted);
  stroke-dasharray: 6 6;
}

.legend,
.decline,
.optimizer,
.kept-name,
.chart .tag {
  color: var(--color-muted);
}

.marker {
  color: var(--color-negative);
}

.swatch,
.bar {
  fill: currentColor;
}

.loss {
  color: var(--color-negative);
}

.cost {
  color: var(--color-ink);
}

.number {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: currentColor;
}
</style>
