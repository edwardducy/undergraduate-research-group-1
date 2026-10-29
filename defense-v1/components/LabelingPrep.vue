<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw } from '../shared/motion'

const slot = { width: 250, height: 230, top: 70 }
const slots = [5, 315, 625]
const middle = slot.top + slot.height / 2
const steps = [
  { name: 'Draft guidelines', detail: 'with Taglish examples', kind: 'guide' },
  { name: 'Set up doccano', detail: 'Labeling tool', kind: 'tool' },
  { name: 'Label a practice batch', detail: 'First run with the guidelines', kind: 'batch' },
]
const guideLines = [36, 48, 60, 72, 84]
const app = { x: slots[1] + 45, y: slot.top + 22, width: 160, height: 94 }
const batch = Array.from({ length: 4 }, (_, index) => ({
  x: slots[2] + 30 + (index % 2) * 100,
  y: slot.top + 22 + Math.floor(index / 2) * 52,
  span: 20 + (index * 7) % 20,
}))
const faded = 0.3

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.step, .link, .legend', { opacity: 0 })
  gsap.set('.link-line', { scaleX: 0, transformOrigin: '0% 50%' })

  timeline.addLabel('step0')
  for (let index = 0; index < steps.length; index++) {
    if (index > 0) {
      timeline
        .to(`.step-${index - 1}`, { opacity: faded, ...exit })
        .to(`.link-${index}`, { opacity: 1, duration: 0 })
        .to(`.link-${index} .link-line`, { scaleX: 1, ...draw })
        .to(`.link-${index} .link-head`, { opacity: 1, duration: 0.1 }, '>-0.05')
    }
    timeline.to(`.step-${index}`, { opacity: 1, ...enter }, index > 0 ? '<' : '>')
    if (index === steps.length - 1) {
      timeline
        .to('.step', { opacity: 1, ...enter }, '<')
        .to('.legend', { opacity: 1, ...enter }, '>')
    }
    timeline.addLabel(`step${index + 1}`)
  }
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <text class="bracket-label" x="440" y="30" text-anchor="middle">Runs alongside the pilot search</text>
    <path class="bracket" d="M 5 50 v -8 h 870 v 8" />
    <rect v-for="x in slots" :key="x" class="slot" :x="x" :y="slot.top" :width="slot.width" :height="slot.height" rx="14" />

    <g v-for="index in [1, 2]" :key="index" class="link" :class="`link-${index}`">
      <path class="link-line" :d="`M ${slots[index - 1] + slot.width + 4} ${middle} H ${slots[index] - 14}`" />
      <path class="link-head" :d="`M ${slots[index] - 4} ${middle} l -11 -6 v 12 z`" />
    </g>

    <g v-for="(step, index) in steps" :key="step.name" class="step" :class="[`step-${index}`, step.kind]">
      <rect class="frame" :x="slots[index]" :y="slot.top" :width="slot.width" :height="slot.height" rx="14" />

      <template v-if="step.kind === 'guide'">
        <rect class="page" :x="slots[0] + 24" :y="slot.top + 22" width="62" height="80" rx="4" />
        <path v-for="(y, row) in guideLines" :key="y" class="page-line" :d="`M ${slots[0] + 34} ${slot.top + y} H ${slots[0] + (row % 2 ? 66 : 76)}`" />
        <rect class="bubble" :x="slots[0] + 100" :y="slot.top + 30" width="132" height="64" rx="8" />
        <text class="bubble-label" :x="slots[0] + 110" :y="slot.top + 50">Example</text>
        <text class="bubble-text" :x="slots[0] + 110" :y="slot.top + 68">"Baha na dito,</text>
        <text class="bubble-text" :x="slots[0] + 110" :y="slot.top + 84">need ng tubig!"</text>
      </template>

      <template v-if="step.kind === 'tool'">
        <rect class="window" :x="app.x" :y="app.y" :width="app.width" :height="app.height" rx="6" />
        <path class="window-rule" :d="`M ${app.x} ${app.y + 16} h ${app.width}`" />
        <circle v-for="dot in 3" :key="dot" class="window-dot" :cx="app.x + 1 + dot * 8" :cy="app.y + 8" r="2.5" />
        <text class="window-name" :x="app.x + app.width / 2" :y="app.y + 12" text-anchor="middle">doccano</text>
        <path class="window-text" :d="`M ${app.x + 12} ${app.y + 34} h 136 M ${app.x + 12} ${app.y + 46} h 108`" />
        <rect v-for="chip in 3" :key="chip" class="chip" :x="app.x + 12 + (chip - 1) * 46" :y="app.y + 62" width="40" height="16" rx="8" />
      </template>

      <template v-if="step.kind === 'batch'">
        <g v-for="(card, row) in batch" :key="row">
          <rect class="post" :x="card.x" :y="card.y" width="90" height="44" rx="6" />
          <circle class="post-mark" :cx="card.x + 11" :cy="card.y + 12" r="5" />
          <path class="post-text" :d="`M ${card.x + 8} ${card.y + 27} h 52 M ${card.x + 8} ${card.y + 35} h 42`" />
          <rect class="span" :x="card.x + card.span" :y="card.y + 23" width="24" height="8" rx="2" />
          <rect class="class-tag" :x="card.x + 66" :y="card.y + 6" width="18" height="12" rx="3" />
        </g>
      </template>

      <text class="step-name" :x="slots[index] + slot.width / 2" :y="slot.top + slot.height - 46" text-anchor="middle">{{ step.name }}</text>
      <text class="step-detail" :x="slots[index] + slot.width / 2" :y="slot.top + slot.height - 22" text-anchor="middle">{{ step.detail }}</text>
    </g>

    <g class="legend">
      <rect class="class-tag" :x="slots[2]" y="321" width="14" height="10" rx="2" />
      <text class="legend-text" :x="slots[2] + 20" y="330">Class label</text>
      <rect class="span" :x="slots[2] + 110" y="321" width="20" height="10" rx="2" />
      <text class="legend-text" :x="slots[2] + 136" y="330">Entity span</text>
    </g>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.bracket-label {
  font-size: 0.95rem;
  fill: var(--color-subtle);
}

.bracket,
.slot {
  fill: none;
  stroke: var(--color-faint);
  stroke-width: 1.5;
}

.slot {
  stroke-dasharray: 6 5;
}

.link-line {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.link-head {
  fill: var(--color-subtle);
}

.frame {
  fill: var(--color-surface);
  stroke-width: 2.5;
}

.guide .frame {
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.tool .frame {
  stroke: var(--color-coding);
}

.batch .frame {
  stroke: var(--color-data);
}

.page {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.page-line {
  stroke: var(--color-faint);
  stroke-width: 2;
}

.bubble {
  fill: #fafafa;
  stroke: var(--color-faint);
}

.bubble-label {
  font-size: 0.75rem;
  fill: var(--color-subtle);
}

.bubble-text {
  font-size: 0.8rem;
  font-style: italic;
  fill: var(--color-muted);
}

.window {
  fill: var(--color-surface);
  stroke: var(--color-coding);
  stroke-width: 1.8;
}

.window-rule {
  stroke: var(--color-coding);
  stroke-width: 1.5;
}

.window-dot {
  fill: var(--color-coding);
}

.window-name {
  font-size: 0.6rem;
  fill: var(--color-coding);
}

.window-text {
  stroke: var(--color-faint);
  stroke-width: 3;
}

.chip {
  fill: none;
  stroke: var(--color-coding);
  stroke-width: 1.4;
}

.post {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
  stroke-width: 1.5;
}

.post-mark {
  fill: var(--color-data);
}

.post-text {
  stroke: var(--color-data);
  stroke-width: 2;
  opacity: 0.4;
}

.span {
  fill: none;
  stroke: var(--color-ner);
  stroke-width: 1.6;
}

.class-tag {
  fill: var(--color-classification);
}

.step-name {
  font-size: 18px;
  font-weight: 700;
  fill: var(--color-ink);
}

.step-detail {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.tool .step-detail {
  fill: var(--color-coding);
}

.batch .step-detail {
  fill: var(--color-data);
}

.legend-text {
  font-size: 0.8rem;
  fill: var(--color-subtle);
}
</style>
