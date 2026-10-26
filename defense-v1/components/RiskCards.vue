<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Laid out like the specific objectives slide. The risks fill in one per click on the left,
// and each brings a drawing of its backup plan on the right

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

// Marks a passed check and the buffer month, the same green as on the schedule slide
const safe = '#15803d'

const risks = [
  { key: 'bugs', name: 'Bugs in the code', plan: 'Each part is tested first, then the whole pipeline' },
  { key: 'memory', name: 'Not enough memory', plan: 'The embedding layer is frozen, so every model fits in 8 GB' },
  { key: 'prototype', name: 'Prototype failures', plan: 'Each failed use case is fixed and tested again' },
  { key: 'data', name: 'Too few posts, or delays', plan: 'Facebook posts fill the gap, and delays use the month of buffer' },
]

const width = 444

// An arrow from one point to another, as a shaft and a head
function arrow(x1: number, y1: number, x2: number, y2: number) {
  const angle = Math.atan2(y2 - y1, x2 - x1)
  const back = { x: x2 - 12 * Math.cos(angle), y: y2 - 12 * Math.sin(angle) }
  const side = { x: -Math.sin(angle) * 7, y: Math.cos(angle) * 7 }
  return {
    shaft: `M ${x1} ${y1} L ${back.x} ${back.y}`,
    head: `M ${x2} ${y2} L ${back.x + side.x} ${back.y + side.y} L ${back.x - side.x} ${back.y - side.y} Z`,
  }
}

// Risk 1. Each part on small examples, then the whole pipeline on a small sample
const parts = ['Data', 'Model', '13 optimizers'].map((name, index) => ({ name, x: 6 + index * 148 }))
const bugArrows = [arrow(222, 150, 222, 206), arrow(222, 338, 222, 368)]

// Risk 2. Illustration only. Peak memory of each of the 13 optimizers in the pilot, in pixels
// above the base. Four go over the 8 GB line, and freezing the embedding layer brings all under it
const chart = { base: 396, limit: 200 }
const bars = [
  [150, 136], [172, 156], [244, 180], [140, 128], [186, 166], [264, 188], [158, 142],
  [232, 172], [130, 118], [176, 158], [280, 192], [164, 148], [190, 170],
].map(([before, after], index) => ({ before, after, x: 6 + index * 33.6, over: before > chart.limit }))

// Risk 3. Test, fix, and test again, centred in the panel
const loop = [
  { name: 'Test', x: 0 },
  { name: 'Fix', x: 160 },
  { name: 'Test again', x: 320 },
]
const loopY = 200
const loopArrows = [arrow(126, loopY + 28, 156, loopY + 28), arrow(286, loopY + 28, 316, loopY + 28)]

// Risk 4. The fallbacks for posts, then the month of buffer
const fallbacks = [
  { name: 'X API', y: 62, last: false },
  { name: 'Public Facebook posts', y: 156, last: false },
  { name: 'Smaller dataset, reported as a limitation', y: 250, last: true },
]
const fallbackArrows = [arrow(222, 120, 222, 152), arrow(222, 214, 222, 246)]
const months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'].map((name, index) => ({ name, x: index * 74, buffer: name === 'Mar' }))

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  // The risks wait in a faint grey until their turn
  gsap.set('.item, .item .plan', { color: colour('faint') })
  gsap.set('.panel, .check, .reveal, .before, .fixed-note', { opacity: 0 })
  gsap.set('.check', { scale: 0.4, transformOrigin: '50% 50%' })
  gsap.set('.memory-bar', { attr: { y: chart.base, height: 0 } })

  let previous = -1
  const focus = (index: number) => {
    if (previous >= 0) {
      timeline
        .to(`.item-${previous}, .item-${previous} .plan`, { color: colour('faint'), ...exit })
        .to(`.panel.${risks[previous].key}`, { opacity: 0, ...exit }, '<')
    }
    timeline
      .to(`.item-${index}`, { color: colour('ink'), ...enter }, '>')
      .to(`.item-${index} .plan`, { color: colour('muted'), ...enter }, '<')
      .to(`.panel.${risks[index].key}`, { opacity: 1, ...enter }, '<')
    previous = index
  }

  timeline.addLabel('step0')

  // Bugs. Each part passes on small examples, then the whole pipeline, then training starts
  focus(0)
  timeline
    .to('.bugs .part-check', { opacity: 1, scale: 1, ...pop, stagger: 0.2 }, '>0.1')
    .to('.bugs .reveal', { opacity: 1, ...enter, stagger: 0.2 }, '>0.1')
    .to('.bugs .whole-check', { opacity: 1, scale: 1, ...pop }, '<0.2')
    .addLabel('step1')

  // Memory. Four optimizers went over the 8 GB graphics card in the pilot
  focus(1)
  timeline
    .to('.memory-bar', { attr: { y: (index: number) => chart.base - bars[index].before, height: (index: number) => bars[index].before }, ...draw, stagger: 0.03 }, '<0.1')
    .addLabel('step2')

  // Freezing the embedding layer brings every optimizer under the line.
  // Where the four were before stays as a dashed outline
    .to('.found', { opacity: 0, ...exit })
    .to('.before', { opacity: 1, ...enter }, '<')
    .to('.memory-bar', { attr: { y: (index: number) => chart.base - bars[index].after, height: (index: number) => bars[index].after }, fill: colour('subtle'), ...draw }, '<')
    .to('.fixed-note', { opacity: 1, ...enter, stagger: 0.15 }, '<0.2')
    .addLabel('step3')

  // Prototype. A failed use case is fixed and tested again
  focus(2)
  timeline
    .to('.prototype .reveal', { opacity: 1, ...enter, stagger: 0.15 }, '>0.1')
    .to('.prototype .check', { opacity: 1, scale: 1, ...pop }, '>')
    .addLabel('step4')

  // Data and delays. The fallbacks for posts, then the month of buffer
  focus(3)
  timeline
    .to('.data .reveal', { opacity: 1, ...enter, stagger: 0.2 }, '>0.1')
    .to('.buffer', { fill: safe, fillOpacity: 0.16, stroke: safe, ...enter })
    .to('.buffer-text', { fill: safe, fontWeight: 700, ...enter }, '<')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="risks">
    <section>
      <h3>Technical risks and backup plans</h3>
      <ol>
        <li v-for="(risk, index) in risks" :key="risk.key" class="item" :class="`item-${index}`">
          <b>{{ index + 1 }}</b>
          <div>
            <p class="name">{{ risk.name }}</p>
            <p class="plan">{{ risk.plan }}</p>
          </div>
        </li>
      </ol>
    </section>

    <svg :viewBox="`0 0 ${width} 456`" :width="width" height="456">
      <!-- Risk 1. Bugs in the code -->
      <g class="panel bugs">
        <text class="tag" :x="width / 2" y="40" text-anchor="middle">Small examples with known answers</text>
        <g v-for="part in parts" :key="part.name">
          <rect class="box" :x="part.x" y="70" width="136" height="64" rx="10" />
          <text class="box-text" :x="part.x + 68" y="109" text-anchor="middle">{{ part.name }}</text>
          <g class="check part-check" :transform="`translate(${part.x + 128} 74)`">
            <circle r="13" />
            <path d="M -6 0 L -1.5 5 L 6.5 -5" />
          </g>
        </g>
        <g class="reveal">
          <path class="line" :d="bugArrows[0].shaft" />
          <path class="head" :d="bugArrows[0].head" />
          <text class="tag" :x="width / 2" y="240" text-anchor="middle">Then a small sample</text>
        </g>
        <g class="reveal">
          <rect class="box" x="42" y="262" width="360" height="72" rx="10" />
          <text class="box-text" :x="width / 2" y="305" text-anchor="middle">Whole pipeline</text>
        </g>
        <g class="check whole-check" transform="translate(394 266)">
          <circle r="13" />
          <path d="M -6 0 L -1.5 5 L 6.5 -5" />
        </g>
        <g class="reveal">
          <path class="line" :d="bugArrows[1].shaft" />
          <path class="head" :d="bugArrows[1].head" />
          <rect class="soft" x="122" y="372" width="200" height="52" rx="10" />
          <text class="box-text" :x="width / 2" y="404" text-anchor="middle">Training starts</text>
        </g>
      </g>

      <!-- Risk 2. Not enough memory -->
      <g class="panel memory">
        <text class="tag" x="0" y="40">Memory of each optimizer in the pilot</text>
        <path class="axis" :d="`M 0 ${chart.base} H ${width}`" />
        <template v-for="(bar, index) in bars" :key="index">
          <rect v-if="bar.over" class="before" :x="bar.x" :y="chart.base - bar.before" width="22" :height="bar.before" rx="3" />
          <rect class="memory-bar" :class="{ over: bar.over }" :x="bar.x" width="22" rx="3" />
        </template>
        <path class="limit" :d="`M 0 ${chart.base - chart.limit} H ${width}`" />
        <text class="limit-text" :x="width" :y="chart.base - chart.limit - 10" text-anchor="end">8 GB</text>
        <text class="note found" x="0" y="432">Four went over the 8 GB card</text>

        <g class="fixed-note">
          <rect class="soft framed" x="0" y="64" width="270" height="56" rx="10" />
          <text class="box-text" x="118" y="99" text-anchor="middle">Embedding layer</text>
          <g class="lock" transform="translate(244 88)">
            <path d="M -6 -2 V -7 A 6 6 0 0 1 6 -7 V -2" />
            <rect x="-10" y="-2" width="20" height="14" rx="3" />
          </g>
          <text class="small" x="286" y="90">Frozen in</text>
          <text class="small" x="286" y="112">every model</text>
        </g>
        <text class="small fixed-note" x="0" y="432">All fit now</text>
      </g>

      <!-- Risk 3. Prototype failures -->
      <g class="panel prototype">
        <text class="tag" x="0" :y="loopY - 24">A use case fails</text>
        <g v-for="(step, index) in loop" :key="step.name" class="reveal">
          <rect class="box" :x="step.x" :y="loopY" width="124" height="56" rx="10" />
          <text class="box-text" :x="step.x + 62" :y="loopY + 35" text-anchor="middle">{{ step.name }}</text>
          <template v-if="index > 0">
            <path class="line" :d="loopArrows[index - 1].shaft" />
            <path class="head" :d="loopArrows[index - 1].head" />
          </template>
        </g>
        <g class="check" :transform="`translate(436 ${loopY + 2})`">
          <circle r="13" />
          <path d="M -6 0 L -1.5 5 L 6.5 -5" />
        </g>
      </g>

      <!-- Risk 4. Too few posts, or delays -->
      <g class="panel data">
        <text class="tag" x="0" y="40">If there are too few Taglish posts</text>
        <g v-for="(fallback, index) in fallbacks" :key="fallback.name" class="reveal">
          <template v-if="index > 0">
            <path class="line" :d="fallbackArrows[index - 1].shaft" />
            <path class="head" :d="fallbackArrows[index - 1].head" />
          </template>
          <rect :class="fallback.last ? 'soft' : 'box'" x="0" :y="fallback.y" :width="width" height="56" rx="10" />
          <text :class="fallback.last ? 'small' : 'box-text'" :x="width / 2" :y="fallback.y + 35" text-anchor="middle">{{ fallback.name }}</text>
        </g>
        <g class="reveal">
          <text class="tag" x="0" y="356">Any other delay</text>
          <g v-for="month in months" :key="month.name">
            <rect class="month" :class="{ buffer: month.buffer }" :x="month.x" y="372" width="70" height="40" rx="6" />
            <text class="small" :class="{ 'buffer-text': month.buffer }" :x="month.x + 35" y="398" text-anchor="middle">{{ month.name }}</text>
          </g>
        </g>
        <text class="buffer-label reveal" x="333" y="440" text-anchor="middle">Buffer</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.risks {
  display: flex;
  align-items: center;
  gap: 56px;
  height: 456px;
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
  gap: var(--space-2);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: grid;
  grid-template-columns: 2rem 1fr;
  line-height: var(--leading-list);
}

li b {
  font-weight: 700;
}

.name {
  margin: 0;
  font-weight: 700;
  line-height: var(--leading-list);
}

.plan {
  margin: 2px 0 0;
  font-size: var(--text-xs);
  line-height: var(--leading-list);
}

svg {
  flex: none;
  overflow: visible;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.box {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.soft {
  fill: var(--color-land);
}

.soft.framed {
  stroke: var(--color-ink);
  stroke-width: 2;
}

.box-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.small,
.note {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
}

.note {
  font-weight: 600;
  fill: var(--color-negative);
}

.line,
.axis {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.axis {
  stroke: var(--color-subtle);
}

.head {
  fill: var(--color-ink);
}

.check circle {
  fill: #15803d;
}

.check path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.memory-bar {
  fill: var(--color-ink);
}

.memory-bar.over {
  fill: var(--color-negative);
}

.before {
  fill: none;
  stroke: var(--color-negative);
  stroke-width: 2;
  stroke-dasharray: 5 4;
}

.limit {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 7 6;
}

.limit-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.lock {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-linecap: round;
}

.month {
  fill: var(--color-land);
  stroke: none;
  stroke-width: 2;
}

.buffer-label {
  font-size: var(--text-2xs);
  font-weight: 700;
  fill: #15803d;
}
</style>
