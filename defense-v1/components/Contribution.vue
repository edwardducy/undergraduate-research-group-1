<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { benchmarkChart as benchmark, benchmarkColumns as columns, benchmarkReach, benchmarkX } from '../shared/benchmark'
import { angle, conflictAngle, easierAngle, harderAngle, model, pivot, reach, shortReach, sideX, strongerReach as fullStrongerReach, tilt, tip } from '../shared/model'
import { enter, exit, draw, moveShort, moveLong } from '../shared/motion'

const transfer = { left: 24, width: 860, top: 64, base: 380 }
const low = 0.95
const high = 1.01
const scoreY = (share: number) => transfer.base - ((transfer.base - transfer.top) * (share - low)) / (high - low)

const groups = [
  {
    lines: ['English', 'text'],
    source: 'Zhu et al. (2025)',
    tasks: [{ kind: 'classification', share: 63.6 / 63.9 }, { kind: 'ner', share: 76.4 / 77.1 }],
  },
  {
    lines: ['Chinese', 'typhoon posts'],
    source: 'Zou et al. (2025)',
    tasks: [{ kind: 'classification', share: 0.875 / 0.876 }, { kind: 'ner', share: 0.860 / 0.854 }],
  },
  { lines: ['Taglish', 'typhoon posts'], source: 'This study', tasks: [] },
].map((group, index) => ({
  ...group,
  x: transfer.left + 30 + ((transfer.width - 30) * (index + 0.5)) / 3,
}))

const spread = 50
const barWidth = 42

const gap = { top: transfer.base - 34, bottom: transfer.base - 24 }
const transferAxis = `M ${transfer.left} ${transfer.top} V ${gap.top} M ${transfer.left} ${gap.bottom} V ${transfer.base} H ${transfer.left + transfer.width}`
const slashes = [gap.top, gap.bottom].map(y => `M ${transfer.left - 8} ${y + 4} L ${transfer.left + 8} ${y - 4}`).join(' ')

const swatch = 14
const stripes = [0.5, 1, 1.5].map((at) => {
  const c = swatch * at
  return c <= swatch ? `M 0 ${c} L ${c} 0` : `M ${c - swatch} ${swatch} L ${swatch} ${c - swatch}`
}).join(' ')
const taskLegendX = transfer.left + 280

const sides = [
  { kind: 'equal', name: 'Equal weighting', x: sideX.equal },
  { kind: 'advanced', name: 'Advanced optimizer', x: sideX.advanced },
]

const strongerReach = (shortReach * fullStrongerReach) / reach
const lean = tip(36, conflictAngle)

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  sides.forEach(side => gsap.set(`.${side.kind}`, { x: side.x }))
  gsap.set('.head', { y: shortReach })
  gsap.set('.pull-classification .arm', { rotation: angle, svgOrigin: '0 0' })
  gsap.set('.pull-ner .arm', { rotation: -angle, svgOrigin: '0 0' })
  gsap.set('.midline, .name-tag, .name, .transfer-text, .benchmark-text, .harder-problem, .easier-problem, .gama, .kurin, .model-ghost', { opacity: 0 })
  gsap.set('.chart-axis, .model', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.chart-bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.pull', { clipPath: 'inset(0% 0% 100% 0%)' })

  const settle = (length: number, move: gsap.TweenVars) => {
    timeline
      .to('.pull-classification .arm', { rotation: angle, svgOrigin: '0 0', ...move }, '<')
      .to('.pull-ner .arm', { rotation: -angle, svgOrigin: '0 0', ...move }, '<')
      .to('.shaft', { attr: { y2: length - 14 }, ...move }, '<')
      .to('.head', { y: length, ...move }, '<')
  }

  timeline
    .addLabel('step0')
    .to('.transfer .chart-axis', { strokeDashoffset: 0, ...draw })
    .to('.transfer-text', { opacity: 1, ...enter }, '<0.2')
    .to('.transfer .chart-bar', { scaleY: 1, ...draw, stagger: 0.08 }, '<')
    .addLabel('step1')

  timeline
    .to('.transfer-text, .transfer .chart-bar, .transfer .chart-axis', { opacity: 0, ...exit })
    .to('.model', { strokeDashoffset: 0, ...draw }, '>')
    .to('.name, .midline, .name-tag', { opacity: 1, ...enter }, '<0.2')
    .fromTo('.pull', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '>-0.1')
    .addLabel('step2')

  timeline
    .to('.pull-classification .arm', { rotation: harderAngle, svgOrigin: '0 0', ...moveShort })
    .to('.pull-ner .arm', { rotation: -harderAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.equal .pull-ner .shaft', { attr: { y2: strongerReach - 14 }, ...moveShort }, '>')
    .to('.equal .pull-ner .head', { y: strongerReach, ...moveShort }, '<')
    .to('.equal .rig', { x: lean.x, y: lean.y, ...moveShort }, '<')
    .to('.equal .model-box', { rotation: tilt, svgOrigin: `0 ${model.middle}`, ...moveShort }, '<')
    .to('.equal .pulls', { x: pivot.x, y: pivot.y, ...moveShort }, '<')
    .to('.harder-problem, .gama', { opacity: 1, ...enter }, '<0.3')
    .to('.equal .model-ghost', { opacity: 1, ...enter }, '<')
    .addLabel('step3')

  timeline
    .to('.harder-problem, .gama', { opacity: 0, ...exit })
    .to('.equal .model-ghost', { opacity: 0, ...exit }, '<')
    .to('.pull-classification .arm', { rotation: easierAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.pull-ner .arm', { rotation: -easierAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.equal .pull-ner .shaft', { attr: { y2: shortReach - 14 }, ...moveShort }, '<')
    .to('.equal .pull-ner .head', { y: shortReach, ...moveShort }, '<')
    .to('.equal .rig', { x: 0, y: 0, ...moveShort }, '<')
    .to('.equal .model-box', { rotation: 0, svgOrigin: `0 ${model.middle}`, ...moveShort }, '<')
    .to('.equal .pulls', { x: 0, y: 0, ...moveShort }, '<')
    .to('.easier-problem, .kurin', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step4')

  timeline
    .to('.easier-problem, .kurin, .name-tag', { opacity: 0, ...exit })
    .to('.side', { x: benchmarkX, ...moveLong }, '<')
  settle(benchmarkReach, moveLong)
  timeline
    .set('.advanced', { opacity: 0 })
    .to('.benchmark .chart-axis', { strokeDashoffset: 0, ...draw }, '>')
    .to('.benchmark-text', { opacity: 1, ...enter }, '<0.2')
    .to('.benchmark .chart-bar.loss', { scaleY: 1, ...draw, stagger: 0.03 }, '<')
    .to('.benchmark .chart-bar.cost', { scaleY: 1, ...draw, stagger: 0.03 }, '<0.2')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="contribution">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <defs>
        <pattern v-for="kind in ['classification', 'ner']" :id="`contribution-hatch-${kind}`" :key="kind" :class="kind" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="8" />
        </pattern>
        <clipPath id="contribution-swatch">
          <rect :width="swatch" :height="swatch" rx="3" />
        </clipPath>
      </defs>

      <g v-for="side in sides" :key="side.kind" class="side" :class="side.kind">
        <path class="midline" d="M 0 56 V 390" />
        <text class="tag name-tag" x="0" y="36" text-anchor="middle">{{ side.name }}</text>

        <rect class="model-ghost" x="-128" y="112" width="256" height="64" rx="12" pathLength="624" />
        <g class="rig">
          <g class="model-box">
            <rect class="model" x="-128" y="112" width="256" height="64" rx="12" pathLength="1" />
            <text class="name" x="0" y="152" text-anchor="middle">Multi-task model</text>
          </g>

          <g class="pulls">
            <g v-for="kind in ['classification', 'ner']" :key="kind" class="pull" :class="[`pull-${kind}`, kind]" :transform="`translate(0 ${model.bottom})`">
              <g class="arm">
                <line class="shaft" x1="0" y1="0" x2="0" :y2="shortReach - 14" />
                <path class="head" d="M 0 0 l -6 -16 h 12 z" />
              </g>
            </g>
          </g>
        </g>
      </g>

      <g class="transfer">
        <g class="transfer-text muted">
          <g class="hatched" :transform="`translate(${transfer.left} ${transfer.top - 50})`">
            <path :d="stripes" clip-path="url(#contribution-swatch)" />
            <rect :width="swatch" :height="swatch" rx="3" />
          </g>
          <text class="tag" :x="transfer.left + 22" :y="transfer.top - 37">Single-task models</text>
          <rect class="swatch" :x="transfer.left" :y="transfer.top - 26" :width="swatch" :height="swatch" rx="3" />
          <text class="tag" :x="transfer.left + 22" :y="transfer.top - 13">Multi-task model</text>

          <g class="classification">
            <rect class="swatch" :x="taskLegendX" :y="transfer.top - 50" :width="swatch" :height="swatch" rx="3" />
            <text class="tag" :x="taskLegendX + 22" :y="transfer.top - 37">Classification</text>
          </g>
          <g class="ner">
            <rect class="swatch" :x="taskLegendX" :y="transfer.top - 26" :width="swatch" :height="swatch" rx="3" />
            <text class="tag" :x="taskLegendX + 22" :y="transfer.top - 13">NER</text>
          </g>

          <text class="tag" text-anchor="middle" :transform="`translate(${transfer.left - 16} ${(transfer.top + transfer.base) / 2}) rotate(-90)`">Higher is better</text>
        </g>

        <g v-for="group in groups" :key="group.source">
          <text class="transfer-text tag group-name" text-anchor="middle">
            <tspan v-for="(line, index) in group.lines" :key="line" :x="group.x" :y="transfer.base + 30 + index * 22">{{ line }}</tspan>
          </text>
          <text class="transfer-text source" :x="group.x" :y="transfer.base + 76" text-anchor="middle">{{ group.source }}</text>

          <g v-for="(task, index) in group.tasks" :key="task.kind" :class="task.kind">
            <rect class="chart-bar single" :x="group.x + (index ? spread : -spread) - barWidth - 2" :y="scoreY(1)" :width="barWidth" :height="transfer.base - scoreY(1)" :fill="`url(#contribution-hatch-${task.kind})`" />
            <rect class="chart-bar" :x="group.x + (index ? spread : -spread) + 2" :y="scoreY(task.share)" :width="barWidth" :height="transfer.base - scoreY(task.share)" />
          </g>
          <text v-if="!group.tasks.length" class="transfer-text unknown" :x="group.x" :y="(scoreY(1) + transfer.base) / 2 + 16" text-anchor="middle">?</text>
        </g>

        <path class="chart-axis" :d="transferAxis" pathLength="1" />
        <path class="transfer-text axis-break" :d="slashes" />
      </g>

      <g class="benchmark">
        <g class="benchmark-text muted">
          <rect class="swatch loss" :x="benchmark.left" :y="benchmark.top - 50" :width="swatch" :height="swatch" rx="3" />
          <text class="tag" :x="benchmark.left + 22" :y="benchmark.top - 37">Negative transfer</text>
          <rect class="swatch cost" :x="benchmark.left" :y="benchmark.top - 26" :width="swatch" :height="swatch" rx="3" />
          <text class="tag" :x="benchmark.left + 22" :y="benchmark.top - 13">Training time and memory</text>
          <text class="tag" text-anchor="middle" :transform="`translate(${benchmark.left - 16} ${(benchmark.top + benchmark.base) / 2}) rotate(-90)`">Lower is better</text>
          <text class="tag" :x="benchmark.left + benchmark.width / 2" y="424" text-anchor="middle">Multi-task optimizer</text>
        </g>

        <g v-for="(column, index) in columns" :key="index">
          <rect class="chart-bar loss" :x="column.x - 15" :y="benchmark.base - column.loss" width="14" :height="column.loss" />
          <rect class="chart-bar cost" :x="column.x + 1" :y="benchmark.base - column.cost" width="14" :height="column.cost" />
          <text class="benchmark-text number" :x="column.x" :y="benchmark.base + 28" text-anchor="middle">{{ index + 1 }}</text>
        </g>

        <path class="chart-axis" :d="`M ${benchmark.left} ${benchmark.top} V ${benchmark.base} H ${benchmark.left + benchmark.width}`" pathLength="1" />
      </g>

      <text class="tag problem harder-problem" x="442" y="424" text-anchor="middle">Harder problem</text>
      <text class="tag problem easier-problem" x="442" y="424" text-anchor="middle">Easier problem</text>
    </svg>

    <footer class="gama">Sources: Kurin et al. (2022), Gama and Grassi (2026)</footer>
    <footer class="kurin">Source: Kurin et al. (2022)</footer>
  </div>
</template>

<style scoped>
.contribution {
  position: relative;
  height: 456px;
}

svg {
  overflow: visible;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: currentColor;
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
  fill: var(--color-ink);
}

.group-name {
  color: var(--color-ink);
}

.source {
  font-size: 0.95rem;
  fill: var(--color-subtle);
}

.unknown {
  font-size: 3rem;
  font-weight: 700;
  fill: var(--color-ink);
}

.number {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
}

.muted,
.name-tag,
.problem {
  color: var(--color-muted);
}

.classification {
  color: var(--color-classification);
}

.ner {
  color: var(--color-ner);
}

.loss {
  color: var(--color-negative);
}

.cost,
.chart-axis,
.axis-break {
  color: var(--color-ink);
}

pattern line {
  stroke: currentColor;
  stroke-width: 3;
}

.chart-bar:not(.single),
.swatch {
  fill: currentColor;
}

.chart-bar.single {
  stroke: currentColor;
  stroke-width: 2;
}

.hatched path {
  stroke-width: 1.5;
}

.hatched rect {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
}

.midline {
  color: var(--color-muted);
  stroke-dasharray: 6 6;
}

.model-ghost {
  fill: none;
  stroke: var(--color-negative);
  stroke-width: 2;
  stroke-dasharray: 6 6;
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

.head {
  fill: currentColor;
  stroke: none;
}

footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
