<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Training pilot. One bar per optimizer. The heights are an illustration, not measured memory
const memory = { width: 273, base: 190, limit: 70 }
const before = [100, 150, 80, 110, 160, 95, 70, 140, 105, 85, 155, 90, 115]
const shrink = 0.72
const bars = before.map((height, index) => ({ x: index * 21 + 3, height, over: memory.base - height < memory.limit }))

// Workload. One cell per run, 92 tuning runs and then 51 main runs
const tuningRuns = 92
const mainRuns = 51
const cell = 14
const pitch = 18
const cells = Array.from({ length: tuningRuns + mainRuns }, (_, index) => ({
  x: (index % 13) * pitch + 20,
  y: Math.floor(index / 13) * pitch,
  main: index >= tuningRuns,
}))

// Budget. The split across cyclones is an illustration until the pilot search is run
const budget = { left: 8, right: 265, top: 72, height: 44 }
const shares = [0.3, 0.15, 0.25, 0.18, 0.12]
const splits = shares.slice(0, -1).map((_, index) => budget.left + (budget.right - budget.left) * shares.slice(0, index + 1).reduce((sum, share) => sum + share, 0))
const centres = shares.map((share, index) => budget.left + (budget.right - budget.left) * (shares.slice(0, index).reduce((sum, value) => sum + value, 0) + share / 2))

// The cyclone symbol from the disaster risk map
const arm = 'M -7 0 C -7 -11 5 -17 17 -15 C 8 -12 3 -8 0 -7 Z'

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveShort = { duration: 0.5, ease: 'power2.inOut' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--color-ink').trim()

  gsap.set('.workload, .budget, .after, .fits, .legend, .cap, .split, .cyclone, .note-budget', { opacity: 0 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.run', { scale: 0, transformOrigin: '50% 50%' })
  gsap.set('.spend', { scaleX: 0, transformOrigin: '0% 50%' })

  timeline
    .addLabel('step0')
    .to('.bar', { scaleY: 1, ...draw, stagger: 0.03 })
    .addLabel('step1')
    .to('.bar', { scaleY: shrink, ...moveShort })
    .to('.bar.over', { color: ink, ...moveShort }, '<')
    .to('.after, .fits', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step2')
    .to('.workload', { opacity: 1, ...enter })
    .to('.run.tuning', { scale: 1, ...pop, stagger: 0.008 })
    .to('.run.main', { scale: 1, ...pop, stagger: 0.008 })
    .to('.legend', { opacity: 1, ...enter }, '<')
    .addLabel('step3')
    .to('.budget', { opacity: 1, ...enter })
    .to('.spend', { scaleX: 1, ...draw }, '<0.2')
    .to('.cap', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step4')
    .to('.split', { opacity: 1, ...enter, stagger: 0.08 })
    .to('.cyclone', { opacity: 1, ...enter, stagger: 0.08 }, '<')
    .to('.note-budget', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="feasibility">
    <div class="panel memory">
      <div class="tag">Training pilot</div>
      <svg :viewBox="`0 0 ${memory.width} 200`">
        <rect v-for="(bar, index) in bars" :key="index" class="bar" :class="{ over: bar.over }" :x="bar.x" :y="memory.base - bar.height" width="14" :height="bar.height" />
        <path class="limit" :d="`M 0 ${memory.limit} H ${memory.width}`" />
        <text class="small" :x="memory.width" :y="memory.limit - 10" text-anchor="end">Laptop memory</text>
        <text class="small after" x="0" y="24">After one change</text>
      </svg>
      <div class="caption fits">Every model fits in laptop memory</div>
    </div>

    <div class="panel workload">
      <div class="tag">Fixed workload</div>
      <svg viewBox="0 0 273 200">
        <rect v-for="(run, index) in cells" :key="index" class="run" :class="run.main ? 'main' : 'tuning'" :x="run.x" :y="run.y" :width="cell" :height="cell" rx="3" />
      </svg>
      <div class="caption legend">
        <span><i class="swatch tuning" />92 tuning runs</span>
        <span><i class="swatch main" />51 main runs</span>
      </div>
    </div>

    <div class="panel budget">
      <div class="tag">X API spending</div>
      <svg viewBox="0 0 273 200">
        <text class="small cap" :x="budget.right" :y="budget.top - 14" text-anchor="end">Capped at 200 US dollars</text>
        <rect class="spend" :x="budget.left" :y="budget.top" :width="budget.right - budget.left" :height="budget.height" rx="6" />
        <rect class="outline" :x="budget.left" :y="budget.top" :width="budget.right - budget.left" :height="budget.height" rx="6" />
        <path v-for="(x, index) in splits" :key="index" class="split" :d="`M ${x} ${budget.top} v ${budget.height}`" />
        <g v-for="(x, index) in centres" :key="index" class="cyclone" :transform="`translate(${x} ${budget.top + budget.height + 32}) scale(0.9)`">
          <path :d="arm" />
          <path :d="arm" transform="rotate(180)" />
          <circle r="7" />
          <circle class="eye" r="2.5" />
        </g>
      </svg>
      <div class="caption note-budget">A pilot search spreads it across the cyclones</div>
    </div>

    <footer>Bar heights and budget shares are illustrations</footer>
  </div>
</template>

<style scoped>
.feasibility {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  column-gap: var(--space-4);
  height: 360px;
}

.panel {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

svg {
  width: 100%;
  height: 200px;
  overflow: visible;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-1);
}

/* A halo keeps the labels readable where a bar runs behind them */
.small {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
  paint-order: stroke;
  stroke: var(--color-surface);
  stroke-width: 6px;
  stroke-linejoin: round;
}

.after {
  fill: var(--color-classification);
}

.caption {
  font-size: var(--text-base);
  font-weight: 600;
  line-height: 1.3;
}

.bar {
  fill: currentColor;
  color: var(--color-ink);
}

.bar.over {
  color: var(--color-negative);
}

.limit {
  stroke: var(--color-muted);
  stroke-width: 2;
  stroke-dasharray: 8 6;
}

.run.tuning,
.swatch.tuning {
  fill: var(--color-subtle);
  background: var(--color-subtle);
}

.run.main,
.swatch.main {
  fill: var(--color-classification);
  background: var(--color-classification);
}

.legend {
  display: flex;
  flex-direction: column;
  gap: var(--space-0);
  font-size: var(--text-2xs);
}

.swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
  margin-right: var(--space-1);
  vertical-align: -1px;
}

.spend {
  fill: var(--color-land);
}

.outline {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.split {
  stroke: var(--color-ink);
  stroke-width: 2;
}

.cyclone {
  fill: var(--color-ink);
}

.cyclone .eye {
  fill: var(--color-surface);
}

/* The note on illustrations sits in the slide's bottom margin, like the sources on the risk map */
footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
