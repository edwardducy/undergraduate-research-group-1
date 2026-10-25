<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

const model = { x: 24, y: 158, width: 200, height: 64 }
const from = { x: model.x + model.width, y: model.y + model.height / 2 }

// Each group of beneficiaries hangs off the model at the height of its name
const targets = [
  {
    key: 'lgu',
    y: 64,
    tag: 'Main beneficiaries',
    name: 'LGU disaster offices',
    notes: ['Report type and key details from one model', 'Can run locally instead of through a paid', 'large language model service', 'Shows what one model gains or gives up'],
  },
  {
    key: 'university',
    y: 238,
    name: 'Local universities',
    notes: ['Training time and memory results show', 'what building such a model needs'],
  },
  {
    key: 'community',
    y: 336,
    name: 'Typhoon-hit communities',
    notes: ['Responders may find urgent requests sooner'],
  },
]

function link(y: number) {
  return `M ${from.x} ${from.y} C ${from.x + 80} ${from.y} ${from.x + 60} ${y} ${from.x + 140} ${y}`
}

const deliverables = [
  { key: 'dataset', name: 'Labeled Taglish dataset', note: 'With its labeling guidelines' },
  { key: 'model', name: 'Final model', note: 'With the benchmark results' },
  { key: 'code', name: 'Code', note: 'Including the prototype web application' },
  { key: 'x', name: 'Posts from X', note: 'Shared as post IDs with labels, as the terms of X require' },
]

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.target, .card, .chips', { opacity: 0 })
  gsap.set('.card', { y: 16 })
  gsap.set('.link', { clipPath: 'inset(0% 100% 0% 0%)' })

  timeline.addLabel('step0')
  targets.forEach(({ key }, index) => {
    timeline
      .to(`.${key} .link`, { clipPath: 'inset(0% 0% 0% 0%)', ...draw })
      .to(`.${key} .target`, { opacity: 1, ...enter }, '<0.3')
      .addLabel(`step${index + 1}`)
  })

  timeline
    .to('.benefits', { opacity: 0, ...exit })
    .to('.card', { opacity: 1, y: 0, ...enter, stagger: 0.1 })
    .addLabel('step4')
    .to('.x .lines', { opacity: 0, ...exit })
    .to('.x .chips', { opacity: 1, ...enter })
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="benefit-map">
    <svg class="benefits" viewBox="0 0 884 380" width="884" height="380">
      <rect class="model" :x="model.x" :y="model.y" :width="model.width" :height="model.height" rx="12" />
      <text class="model-name" :x="model.x + model.width / 2" :y="model.y + 40" text-anchor="middle">Final model</text>

      <g v-for="target in targets" :key="target.key" :class="target.key">
        <g class="link">
          <path :d="link(target.y)" />
        </g>
        <g class="target">
          <circle :cx="from.x + 148" :cy="target.y" r="6" />
          <text v-if="target.tag" class="tag" x="388" :y="target.y - 28">{{ target.tag }}</text>
          <text class="name" x="388" :y="target.y + 8">{{ target.name }}</text>
          <text v-for="(note, index) in target.notes" :key="index" class="note" x="388" :y="target.y + 34 + index * 22">{{ note }}</text>
        </g>
      </g>
    </svg>

    <div class="deliverables">
      <div v-for="item in deliverables" :key="item.key" class="card" :class="item.key">
        <svg class="icon" viewBox="0 0 48 48">
          <template v-if="item.key === 'dataset'">
            <rect x="6" y="8" width="36" height="32" rx="4" />
            <path d="M 6 18 H 42 M 6 28 H 42 M 20 8 V 40" />
          </template>
          <template v-else-if="item.key === 'model'">
            <rect x="6" y="14" width="36" height="20" rx="5" />
            <path d="M 24 4 V 14 M 16 34 V 44 M 32 34 V 44" />
          </template>
          <template v-else-if="item.key === 'code'">
            <path d="M 16 12 L 4 24 L 16 36 M 32 12 L 44 24 L 32 36 M 27 8 L 21 40" />
          </template>
          <template v-else>
            <g class="lines">
              <path d="M 6 14 H 42 M 6 24 H 36 M 6 34 H 30" />
            </g>
            <g class="chips">
              <rect x="4" y="10" width="40" height="12" rx="6" />
              <rect class="label-chip" x="4" y="28" width="26" height="12" rx="6" />
            </g>
          </template>
        </svg>
        <div>
          <div class="name">{{ item.name }}</div>
          <div class="note">{{ item.note }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.benefit-map {
  position: relative;
  height: 380px;
}

.benefits {
  position: absolute;
  inset: 0;
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.model-name,
.name {
  font-size: var(--text-base);
  font-weight: 600;
}

text {
  fill: currentColor;
}

.link path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2.5;
}

.lgu {
  color: var(--color-classification);
}

.lgu .link path {
  stroke-width: 5;
}

.community .link path {
  stroke-dasharray: 8 6;
}

.target circle {
  fill: currentColor;
}

.target .name {
  fill: var(--color-ink);
}

.note {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
  color: var(--color-muted);
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.deliverables {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: var(--space-3);
}

.card {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border: 2px solid var(--color-land);
  border-radius: 12px;
}

.card .note {
  margin-top: var(--space-0);
  line-height: 1.3;
}

.icon {
  flex: none;
  width: 56px;
  height: 56px;
  color: var(--color-classification);
}

.icon rect,
.icon path {
  fill: none;
  stroke: currentColor;
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.icon .lines path {
  stroke: var(--color-subtle);
}

.icon .label-chip {
  stroke: var(--color-ner);
}
</style>
