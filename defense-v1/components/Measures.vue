<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, moveLong, pop } from '../shared/motion'

const lost = 14

// Cost scene. One measure per click, with a 2x2 map of the four measures in the corner.
const heads = [
  { eyebrow: 'Speed · While training', title: 'Training time' },
  { eyebrow: 'Speed · When labeling posts', title: 'Time to label each post' },
  { eyebrow: 'Memory · While training', title: 'Peak memory' },
  { eyebrow: 'Memory · When labeling posts', title: 'Storage size of the saved model' },
]
const mapCells = [0, 1, 2, 3].map(k => ({ k, x: 818 + (k % 2) * 36, y: Math.floor(k / 2) * 36 }))

const filePath = (x: number, y: number, w: number, h: number) => {
  const c = w * 0.25
  return { body: `M ${x} ${y} h ${w - c} l ${c} ${c} v ${h - c} h ${-w} z`, fold: `M ${x + w - c} ${y} v ${c} h ${c}` }
}
const arrowPath = (x1: number, x2: number, y: number) => `M ${x1} ${y} H ${x2} M ${x2 - 9} ${y - 8} l 9 8 l -9 8`
const clocks = [
  { key: 'a', x: 340, y: 385, r: 24, turn: 0.8 },
  { key: 'b', x: 420, y: 135, r: 20, turn: 0.15 },
]

const trainingBlocks = Array.from({ length: 16 }, (_, i) => ({ x: 40 + (i % 4) * 30, y: 150 + Math.floor(i / 4) * 30 }))
const smallFile = filePath(570, 159, 70, 96)
const bigFile = filePath(367, 130, 150, 200)

const queue = [281, 351]
const post = { x: 40, y: 205, w: 200 }
const labeled = { x: 595, y: 205, w: 200 }

const chart = { x0: 40, x1: 760, base: 410, height: 230 }
const memory = [[0, 0], [0.06, 0.45], [0.14, 0.55], [0.22, 0.5], [0.3, 0.62], [0.38, 0.6], [0.46, 0.83], [0.52, 0.9], [0.58, 0.78], [0.66, 0.6], [0.74, 0.64], [0.82, 0.55], [0.9, 0.5], [0.97, 0.3], [1, 0]]
  .map(([x, y]) => [chart.x0 + x * (chart.x1 - chart.x0), chart.base - y * chart.height])
const memoryLine = `M ${memory.map(p => p.join(' ')).join(' L ')}`
const peak = memory[7]
const cardParts = (x: number, w: number) => [0.3, 0.16, 0.4].map((share, i) => ({ left: x + 12 + i * 8 + w * [0, 0.3, 0.46][i], width: w * share }))
const clockTop = (c: { x: number, y: number, r: number }) => `M ${c.x} ${c.y - c.r * 1.3} V ${c.y - c.r} M ${c.x - c.r * 0.25} ${c.y - c.r * 1.33} H ${c.x + c.r * 0.25}`

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.scene.cost, .transfer-part', { opacity: 0 })
  gsap.set('.head, .panel:not(.a), .fade', { opacity: 0 })
  gsap.set('.panel.a', { x: 0 })
  gsap.set('.flow-arrow, .a .bracket, .b .bracket, .peak-line', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.d .bracket', { scaleY: 0, transformOrigin: '50% 0%' })
  gsap.set('.train-block, .file, .output, .peak-dot', { transformOrigin: '50% 50%' })
  gsap.set('.peak-dot', { scale: 0 })
  gsap.set('.memory-line', { strokeDashoffset: 1 })
  gsap.set('.hand', { rotation: 0 })
  gsap.set('.d .big-file', { x: 203, y: 29, scale: 0.47, transformOrigin: '0% 0%' })
  gsap.set('.map-cell', { fill: colour('surface'), stroke: colour('faint') })
  gsap.set('.map-label', { fill: colour('subtle') })
  gsap.set('.multi', { width: '100%' })
  gsap.set('.drop, .dot-value', { opacity: 0 })
  gsap.set('.dot-wrap', { y: 0 })
  gsap.set('.fall', { scaleY: 0, transformOrigin: '50% 0%' })

  timeline
    .addLabel('step0')
    .to('.transfer-part.bars', { opacity: 1, ...enter })
    .to('.multi', { width: `${100 - lost}%`, ...draw })
    .to('.drop', { opacity: 1, ...enter }, '<0.2')
    .to('.transfer-part.axis', { opacity: 1, ...enter }, '<')
    .to('.dot-wrap', { y: 38, ...draw }, '<0.2')
    .to('.fall', { scaleY: 1, ...draw }, '<')
    .to('.dot-value', { opacity: 1, ...enter }, '>-0.1')
    .to('.transfer-part.result', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step1')

  // Each map cell is dark for the current measure, grey once shown, and empty before.
  const showMap = (i: number, at: string) => {
    for (const { k } of mapCells) {
      const tone = k === i ? colour('ink') : k < i ? '#a3a3a3' : ''
      timeline.to(`.map-cell.k${k}`, { fill: tone || colour('surface'), stroke: tone || colour('faint'), ...enter }, at)
    }
    timeline
      .to('.map-label.speed', { fill: colour(i < 2 ? 'ink' : 'subtle'), ...enter }, at)
      .to('.map-label.memory', { fill: colour(i >= 2 ? 'ink' : 'subtle'), ...enter }, at)
  }
  const turnHand = (key: string, at: string) => {
    const clock = clocks.find(c => c.key === key)!
    timeline
      .to(`.clock.${key}`, { opacity: 1, ...enter }, at)
      .to(`.clock.${key} .hand`, { rotation: clock.turn * 360, svgOrigin: `${clock.x} ${clock.y}`, duration: 0.8, ease: 'power1.inOut' }, '<')
  }

  // Click 2. Training posts flow through the model into a saved file, and the whole trip is timed.
  timeline
    .to('.scene.transfer', { opacity: 0, ...exit })
    .to('.scene.cost', { opacity: 1, ...enter })
    .to('.head.h0', { opacity: 1, ...enter }, '<')
  showMap(0, '<')
  timeline
    .fromTo('.a .train-block', { scale: 0.6 }, { opacity: 1, scale: 1, ...pop, stagger: 0.02 }, '<0.1')
    .to('.a .label.posts', { opacity: 1, ...enter }, '<0.2')
    .to('.a .flow-arrow.first', { scaleX: 1, ...draw }, '>-0.1')
    .to('.a .model', { opacity: 1, ...enter }, '>-0.2')
    .to('.a .flow-arrow.second', { scaleX: 1, ...draw }, '>-0.1')
    .fromTo('.a .file', { scale: 0.8 }, { opacity: 1, scale: 1, ...pop }, '>-0.2')
    .to('.a .label.saved', { opacity: 1, ...enter }, '<')
    .to('.a .bracket', { scaleX: 1, ...draw })
  turnHand('a', '>-0.2')
  timeline
    .to('.a .caption', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step2')

  // Click 3. One post goes in and comes out labeled, and only that trip is timed.
    .to('.panel.a', { x: -80, opacity: 0, ...exit })
    .to('.head.h0', { opacity: 0, ...exit }, '<')
    .to('.head.h1', { opacity: 1, ...enter })
  showMap(1, '<')
  timeline
    .to('.panel.b', { opacity: 1, duration: 0 }, '<')
    .fromTo('.b .post', { x: -30 }, { x: 0, opacity: 1, ...enter }, '<')
    .to('.b .queued', { opacity: 1, ...enter, stagger: 0.1 }, '<0.1')
    .to('.b .flow-arrow.first', { scaleX: 1, ...draw }, '>-0.1')
    .to('.b .model', { opacity: 1, ...enter }, '>-0.2')
    .to('.b .flow-arrow.second', { scaleX: 1, ...draw }, '>-0.1')
    .fromTo('.b .output', { scale: 0.8 }, { opacity: 1, scale: 1, ...pop }, '>-0.2')
    .to('.b .bracket', { scaleX: 1, ...draw })
  turnHand('b', '>-0.2')
  timeline
    .to('.b .caption, .b .note', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step3')

  // Click 4. Memory use rises and falls during training, and its highest point is marked.
    .to('.panel.b', { opacity: 0, ...exit })
    .to('.head.h1', { opacity: 0, ...exit }, '<')
    .to('.head.h2', { opacity: 1, ...enter })
  showMap(2, '<')
  timeline
    .to('.panel.c', { opacity: 1, duration: 0 }, '<')
    .to('.c .axis-base, .c .label.start, .c .label.end', { opacity: 1, ...enter }, '<')
    .to('.c .memory-line', { strokeDashoffset: 0, duration: 1.2, ease: 'power1.inOut' })
    .to('.c .memory-area', { opacity: 1, ...enter }, '>-0.3')
    .to('.c .label.in-use', { opacity: 1, ...enter }, '<')
    .to('.c .peak-line', { scaleX: 1, ...draw })
    .to('.c .peak-dot', { scale: 1, ...pop }, '<0.2')
    .to('.c .caption.peak', { opacity: 1, ...enter }, '<')
    .addLabel('step4')

  // Click 5. The saved model file comes back, large and centred, with a bracket for its size.
    .to('.panel.c', { opacity: 0, ...exit })
    .to('.head.h2', { opacity: 0, ...exit }, '<')
    .to('.head.h3', { opacity: 1, ...enter })
  showMap(3, '<')
  timeline
    .to('.panel.d', { opacity: 1, duration: 0 }, '<')
    .to('.d .big-file', { x: 0, y: 0, scale: 1, opacity: 1, ...moveLong }, '<')
    .to('.d .bracket', { scaleY: 1, ...draw }, '>-0.1')
    .to('.d .caption.space, .d .note', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="measures">
    <section class="scene transfer">
      <div class="tag">Accuracy, negative transfer</div>
      <p class="statement">Negative transfer is the multi-task score minus the single-task score</p>
      <div class="transfer-layout">
        <div class="transfer-part bars">
          <div class="pair">
            <span class="who-label">Single-task model</span>
            <div class="long single" />
          </div>
          <div class="pair">
            <span class="who-label">Multi-task model</span>
            <div class="long track">
              <div class="multi" />
              <div class="drop" :style="{ width: `${lost}%` }"><span class="drop-label">Lost accuracy</span></div>
            </div>
          </div>
        </div>
        <div class="transfer-part axis">
          <span class="axis-label gain">Above the line, a gain</span>
          <div class="axis-line" />
          <div class="zero"><span>No change</span></div>
          <div class="fall" />
          <div class="dot-wrap">
            <div class="dot" />
            <span class="dot-value">Lower</span>
          </div>
          <span class="axis-label loss">Below the line, a loss</span>
        </div>
      </div>
      <p class="transfer-part result">The multi-task model scored lower, so this is <strong class="negative">negative transfer</strong></p>
    </section>

  <section class="scene cost">
      <svg viewBox="0 0 884 456" width="884" height="456">
        <g v-for="(head, i) in heads" :key="head.title" class="head" :class="`h${i}`">
          <text class="eyebrow" x="0" y="18">{{ head.eyebrow }}</text>
          <text class="title" x="0" y="64">{{ head.title }}</text>
        </g>
        <g class="map">
          <rect v-for="cell in mapCells" :key="cell.k" class="map-cell" :class="`k${cell.k}`" :x="cell.x" :y="cell.y" width="30" height="30" rx="5" />
          <text class="map-label speed" x="806" y="21" text-anchor="end">Speed</text>
          <text class="map-label memory" x="806" y="57" text-anchor="end">Memory</text>
        </g>

        <g class="panel a">
          <rect v-for="(block, i) in trainingBlocks" :key="i" class="train-block fade" :x="block.x" :y="block.y" width="24" height="24" rx="4" />
          <text class="label posts fade" x="97" y="292" text-anchor="middle">Training posts</text>
          <path class="flow-arrow first" :d="arrowPath(175, 280, 207)" />
          <g class="model fade">
            <rect class="model-box" x="290" y="167" width="150" height="80" rx="12" />
            <text class="model-text" x="365" y="215" text-anchor="middle">Model</text>
          </g>
          <path class="flow-arrow second" :d="arrowPath(455, 555, 207)" />
          <g class="file fade">
            <path class="file-body" :d="smallFile.body" />
            <path class="file-fold" :d="smallFile.fold" />
          </g>
          <text class="label saved fade" x="605" y="285" text-anchor="middle">Saved model</text>
          <path class="bracket" d="M 40 318 V 342 M 40 330 H 640 M 640 318 V 342" />
          <g class="clock a fade">
            <circle :cx="clocks[0].x" :cy="clocks[0].y" :r="clocks[0].r" />
            <path :d="clockTop(clocks[0])" />
            <line class="hand" :x1="clocks[0].x" :y1="clocks[0].y" :x2="clocks[0].x" :y2="clocks[0].y - clocks[0].r * 0.7" />
          </g>
          <text class="caption fade" x="378" y="393">From the first training post to the saved model</text>
        </g>

        <g class="panel b">
          <path class="bracket" d="M 140 160 V 184 M 140 172 H 695 M 695 160 V 184" />
          <g class="clock b fade">
            <circle :cx="clocks[1].x" :cy="clocks[1].y" :r="clocks[1].r" />
            <path :d="clockTop(clocks[1])" />
            <line class="hand" :x1="clocks[1].x" :y1="clocks[1].y" :x2="clocks[1].x" :y2="clocks[1].y - clocks[1].r * 0.7" />
          </g>
          <text class="caption fade" x="455" y="142">One post in, its labels out</text>
          <g class="post fade">
            <rect class="card-box" :x="post.x" :y="post.y" :width="post.w" height="56" rx="8" />
            <rect v-for="part in cardParts(post.x, post.w)" :key="part.left" class="bar" :x="part.left" :y="post.y + 24" :width="part.width" height="8" rx="4" />
          </g>
          <g v-for="y in queue" :key="y" class="queued fade">
            <rect class="card-box" :x="post.x" :y="y" :width="post.w" height="56" rx="8" />
            <rect v-for="part in cardParts(post.x, post.w)" :key="part.left" class="bar" :x="part.left" :y="y + 24" :width="part.width" height="8" rx="4" />
          </g>
          <path class="flow-arrow first" :d="arrowPath(255, 330, 233)" />
          <g class="model fade">
            <rect class="model-box" x="345" y="193" width="150" height="80" rx="12" />
            <text class="model-text" x="420" y="241" text-anchor="middle">Model</text>
          </g>
          <path class="flow-arrow second" :d="arrowPath(505, 580, 233)" />
          <g class="output fade">
            <rect class="card-box" :x="labeled.x" :y="labeled.y" :width="labeled.w" height="56" rx="8" />
            <template v-for="(part, i) in cardParts(labeled.x, labeled.w)" :key="part.left">
              <rect v-if="i === 1" class="bar" :x="part.left" :y="labeled.y + 24" :width="part.width" height="8" rx="4" />
              <rect v-else class="chip" :class="i === 0 ? 'classification' : 'ner'" :x="part.left" :y="labeled.y + 15" :width="part.width" height="26" rx="5" />
            </template>
          </g>
          <text class="note fade" x="40" y="440">The next post waits its turn</text>
        </g>

        <g class="panel c">
          <path class="memory-area fade" :d="`${memoryLine} Z`" />
          <path class="memory-line" :d="memoryLine" pathLength="1" />
          <line class="axis-base fade" :x1="chart.x0" :y1="chart.base" :x2="chart.x1" :y2="chart.base" />
          <text class="label start fade" :x="chart.x0" :y="chart.base + 30">Start of training</text>
          <text class="label end fade" :x="chart.x1" :y="chart.base + 30" text-anchor="end">End</text>
          <text class="label in-use fade" :x="chart.x0 + 90" :y="chart.base - 40">Memory in use</text>
          <line class="peak-line" :x1="chart.x0" :y1="peak[1]" :x2="chart.x1" :y2="peak[1]" />
          <circle class="peak-dot" :cx="peak[0]" :cy="peak[1]" r="10" />
          <text class="caption peak fade" :x="peak[0] + 20" :y="peak[1] - 14">The highest point is recorded</text>
        </g>

        <g class="panel d">
          <g class="big-file fade">
            <path class="file-body" :d="bigFile.body" />
            <path class="file-fold" :d="bigFile.fold" />
            <text class="model-text" x="442" y="260" text-anchor="middle">Model</text>
          </g>
          <path class="bracket" d="M 535 130 h 18 v 200 h -18" />
          <text class="caption space fade" x="571" y="226">Space it takes</text>
          <text class="caption space fade" x="571" y="254">on disk</text>
          <text class="note fade" x="442" y="390" text-anchor="middle">What a computer must store to run it</text>
        </g>
      </svg>
    </section>

  </div>
</template>

<style scoped>
.measures {
  position: relative;
  height: 456px;
}

.scene {
  position: absolute;
  inset: 0;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-1);
}

.statement {
  font-size: var(--text-md);
  font-weight: 600;
  margin: var(--space-1) 0 var(--space-4);
}

.transfer-layout {
  display: grid;
  grid-template-columns: 1fr 200px;
  column-gap: var(--space-5);
  align-items: center;
}

.pair {
  display: grid;
  grid-template-columns: 190px 1fr;
  align-items: center;
  margin-bottom: var(--space-2);
  font-size: var(--text-xs);
}

.long {
  height: 28px;
  border-radius: 4px;
}

.single {
  width: 100%;
  background: var(--color-subtle);
}

.track {
  position: relative;
  width: 100%;
}

.multi {
  height: 100%;
  border-radius: 4px;
  background: var(--color-ink);
}

.drop {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  box-sizing: border-box;
  border: 2px dashed var(--color-negative);
  border-radius: 4px;
  background: color-mix(in srgb, var(--color-negative) 8%, var(--color-surface));
}

.drop-label {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  white-space: nowrap;
  font-size: var(--text-2xs);
  font-weight: 700;
  color: var(--color-ink);
}

.axis {
  position: relative;
  height: 160px;
}

.axis-line {
  position: absolute;
  left: 50%;
  top: 28px;
  bottom: 28px;
  border-left: 2px solid var(--color-ink);
}

.zero {
  position: absolute;
  left: calc(50% - 30px);
  top: 50%;
  width: 60px;
  border-top: 2px solid var(--color-ink);
}

.zero span {
  position: absolute;
  left: calc(100% + 8px);
  top: -12px;
  white-space: nowrap;
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-muted);
}

.fall {
  position: absolute;
  left: calc(50% - 2px);
  top: 50%;
  width: 4px;
  height: 38px;
  background: var(--color-negative);
}

.dot-wrap {
  position: absolute;
  left: calc(50% - 8px);
  top: calc(50% - 8px);
}

.dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-negative);
}

.dot-value {
  position: absolute;
  left: 24px;
  top: -4px;
  white-space: nowrap;
  font-size: var(--text-xs);
  font-weight: 700;
  color: var(--color-ink);
}

.axis-label {
  position: absolute;
  left: 0;
  right: 0;
  text-align: center;
  font-size: var(--text-2xs);
  color: var(--color-muted);
}

.axis-label.gain {
  top: 0;
}

.axis-label.loss {
  bottom: 0;
}

.transfer-part.result {
  margin-top: var(--space-3);
  font-size: var(--text-md);
  font-weight: 700;
  color: var(--color-ink);
}

.transfer-part.result .negative {
  color: var(--color-negative);
}

.eyebrow {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.title {
  font-size: var(--text-3xl);
  font-weight: 700;
  fill: var(--color-ink);
}

.map-cell {
  stroke-width: 2;
}

.map-label,
.label,
.note {
  font-size: var(--text-2xs);
  fill: var(--color-subtle);
}

.caption {
  font-size: var(--text-xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.train-block {
  fill: var(--color-data);
}

.flow-arrow {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 3;
}

.model-box {
  fill: var(--color-land);
  stroke: var(--color-ink);
  stroke-width: 3;
}

.model-text {
  font-size: var(--text-xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.file-body,
.file-fold {
  stroke: var(--color-ink);
  stroke-width: 4;
  stroke-linejoin: round;
}

.file-body {
  fill: var(--color-land);
}

.file-fold,
.bracket {
  fill: none;
}

.bracket {
  stroke: var(--color-ink);
  stroke-width: 3;
}

.clock circle,
.clock path,
.clock line {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 4;
  stroke-linecap: round;
}

.card-box {
  fill: var(--color-surface);
  stroke: var(--color-data);
  stroke-width: 3;
}

.queued .card-box {
  stroke: var(--color-faint);
}

.bar {
  fill: var(--color-faint);
}

.chip {
  stroke-width: 2;
}

.chip.classification {
  fill: color-mix(in srgb, var(--color-classification) 10%, var(--color-surface));
  stroke: var(--color-classification);
}

.chip.ner {
  fill: color-mix(in srgb, var(--color-ner) 12%, var(--color-surface));
  stroke: var(--color-ner);
}

.memory-area {
  fill: var(--color-land);
}

.memory-line {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 4;
  stroke-linejoin: round;
  stroke-dasharray: 1;
}

.axis-base {
  stroke: var(--color-ink);
  stroke-width: 3;
}

.peak-line {
  stroke: var(--color-ink);
  stroke-width: 2.5;
  stroke-dasharray: 9 7;
}

.peak-dot {
  fill: var(--color-ink);
}
</style>
