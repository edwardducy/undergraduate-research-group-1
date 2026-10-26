<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

// Two annotators, each with a copy of the same batch. A label is a category chip at the start
// of a post and a mark under each entity. Illustration only
const annotators = [{ x: 200, name: 'Annotator 1' }, { x: 684, name: 'Annotator 2' }]
const card = { width: 240, height: 176, top: 128 }
const posts = [[[70, 40], [124, 52]], [[70, 60]], [[96, 44], [150, 30]], [[70, 36], [116, 56]]]

// The annotators then move aside to make room for the chart, as the model does on slide 4
const aside = { x: -40, y: 90, scale: 0.56 }

// Agreement on a 0 to 1 scale. The targets come from the script, Cohen's kappa for the
// categories and F1 for the entities. The scores are an illustration: a batch that meets
// both targets, then one that falls short on the entities
const chart = { base: 320, top: 60 }
const y = (value: number) => chart.base - (chart.base - chart.top) * value
const bars = [
  { key: 'categories', name: 'Categories', measure: 'Cohen\'s kappa', x: 560, target: 0.61, first: 0.78, second: 0.75 },
  { key: 'entities', name: 'Entities', measure: 'F1-score', x: 760, target: 0.74, first: 0.83, second: 0.66 },
].map(bar => ({ ...bar, failed: bar.second < bar.target }))
const score = (value: number) => `.${Math.round(value * 100)}`

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const token = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

  gsap.set('.compare, .chart, .bar-group > *:not(.bar), .second, .guidelines', { opacity: 0 })
  gsap.set('.mark', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.bar', { attr: { y: chart.base, height: 0 } })
  gsap.set('.mark-check', { scale: 0.4, transformOrigin: '50% 50%' })

  timeline
    .addLabel('step0')
    // Both annotators label the batch, and the two sets of labels are compared
    .to('.chip', { fill: token('--color-classification'), ...enter, stagger: 0.05 })
    .to('.mark', { scaleX: 1, ...draw, stagger: 0.04 }, '<0.1')
    .to('.compare', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step1')

    // The annotators move aside and the first batch's agreement rises past both targets
    .to('.compare', { opacity: 0, ...exit })
    .to('.scene', { x: aside.x, y: aside.y, scale: aside.scale, svgOrigin: '0 0', ...moveLong }, '<')
    .to('.chart, .target, .target-text, .name, .measure', { opacity: 1, ...enter }, '>-0.2')
    .to('.bar', { attr: { y: (index: number) => y(bars[index].first), height: (index: number) => chart.base - y(bars[index].first) }, ...draw, stagger: 0.1 }, '<0.1')
    .to('.first', { opacity: 1, ...enter, stagger: 0.1 }, '>-0.1')
    .to('.first.mark-check', { scale: 1, ...pop, stagger: 0.1 }, '<')
    .addLabel('step2')

    // The next batch. The entities fall below their target
    .to('.first', { opacity: 0, ...exit })
    .to('.bar', { attr: { y: (index: number) => y(bars[index].second), height: (index: number) => chart.base - y(bars[index].second) }, ...draw }, '<')
    .to('.bar.failed', { fill: token('--color-negative'), ...draw }, '<')
    .to('.second', { opacity: 1, ...enter }, '>-0.1')
    .fromTo('.second.mark-check', { scale: 0.4 }, { scale: 1, ...pop }, '<')
    .addLabel('step3')

    // Before the next batch, the labeling guidelines are revised
    .to('.scene', { opacity: 0, ...exit })
    .to('.guidelines', { opacity: 1, ...enter })
    .fromTo('.pencil', { x: -40, y: -60 }, { x: 0, y: 0, ...moveLong }, '<')
    .addLabel('step4')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 384" width="884" height="384">
    <!-- Two annotators and the same batch, in the middle until the chart arrives -->
    <g class="scene">
      <g v-for="person in annotators" :key="person.name">
        <circle class="person" :cx="person.x" cy="30" r="14" />
        <path class="person" :d="`M ${person.x - 24} 80 C ${person.x - 24} 58 ${person.x - 12} 50 ${person.x} 50 C ${person.x + 12} 50 ${person.x + 24} 58 ${person.x + 24} 80`" />
        <text class="name-text" :x="person.x" y="108" text-anchor="middle">{{ person.name }}</text>
        <rect class="card" :x="person.x - card.width / 2" :y="card.top" :width="card.width" :height="card.height" rx="10" />
        <g v-for="(entities, index) in posts" :key="index" :transform="`translate(${person.x - card.width / 2} ${card.top + 32 + index * 38})`">
          <rect class="chip" x="16" y="-10" width="34" height="20" rx="4" />
          <path class="post-line" d="M 62 0 H 220" />
          <path v-for="[start, length] in entities" :key="start" class="mark" :d="`M ${start - 8} 9 h ${length}`" />
        </g>
      </g>
      <text class="tag" x="442" y="222" text-anchor="middle">Same batch</text>
      <g class="compare">
        <path class="arrow-line" d="M 332 260 H 552" />
        <path class="arrow-head" d="M 332 260 l 12 -7 v 14 z M 552 260 l -12 -7 v 14 z" />
        <text class="name-text" x="442" y="292" text-anchor="middle">How much they agree</text>
      </g>
    </g>

    <!-- Before the next batch, the guidelines are revised -->
    <g class="guidelines">
      <text class="tag" x="0" y="84">Before the next batch</text>
      <rect class="card" x="40" y="110" width="140" height="176" rx="8" />
      <path v-for="line in 6" :key="line" class="post-line" :d="`M 60 ${144 + (line - 1) * 22} H ${160 - ((line - 1) % 3) * 18}`" />
      <g transform="translate(176 250) rotate(-40)">
        <g class="pencil">
          <rect x="-8" y="-60" width="16" height="64" rx="2" />
          <path d="M -8 4 L 0 18 L 8 4 Z" />
        </g>
      </g>
      <text class="name-text" x="110" y="316" text-anchor="middle">Labeling guidelines</text>
      <text class="measure-text" x="110" y="342" text-anchor="middle">are revised</text>
    </g>

    <!-- Agreement after each batch, against its target -->
    <g class="chart">
      <text class="tag first" x="470" y="14">Batch 1</text>
      <text class="tag second" x="470" y="14">Batch 2</text>
      <text class="footer" x="884" y="14" text-anchor="end">Dashed line, the target</text>
      <path class="axis" :d="`M 460 ${chart.base} H 884`" />
    </g>
    <g v-for="bar in bars" :key="bar.key" class="bar-group" :class="bar.key">
      <rect class="bar" :class="{ failed: bar.failed }" :x="bar.x - 40" width="80" rx="3" />
      <path class="target" :d="`M ${bar.x - 70} ${y(bar.target)} H ${bar.x + 70}`" />
      <text class="target-text" :x="bar.x + 74" :y="y(bar.target) + 6">{{ score(bar.target) }}</text>

      <text class="score first" :x="bar.x" :y="y(bar.first) - 34" text-anchor="middle">{{ score(bar.first) }}</text>
      <g class="mark-check first" :transform="`translate(${bar.x} ${y(bar.first) - 72})`">
        <circle class="pass" r="13" />
        <path class="tick" d="M -6 0 L -1.5 5 L 6.5 -5" />
      </g>

      <!-- On the second batch a failed score sits above its target line, so it stays readable -->
      <text class="score second" :class="{ failed: bar.failed }" :x="bar.x" :y="(bar.failed ? y(bar.target) : y(bar.second)) - 34" text-anchor="middle">{{ score(bar.second) }}</text>
      <g class="mark-check second" :transform="`translate(${bar.x} ${(bar.failed ? y(bar.target) : y(bar.second)) - 72})`">
        <circle :class="bar.failed ? 'fail' : 'pass'" r="13" />
        <path v-if="bar.failed" class="tick" d="M -5 -5 L 5 5 M 5 -5 L -5 5" />
        <path v-else class="tick" d="M -6 0 L -1.5 5 L 6.5 -5" />
      </g>

      <text class="name" :x="bar.x" :y="chart.base + 28" text-anchor="middle">{{ bar.name }}</text>
      <text class="measure" :x="bar.x" :y="chart.base + 52" text-anchor="middle">{{ bar.measure }}</text>
    </g>
  </svg>
</template>

<style scoped>
.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.name-text,
.name {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.measure,
.measure-text {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
}

.footer {
  font-size: var(--text-2xs);
  fill: var(--color-subtle);
}

.person {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2.5;
}

.card {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.chip {
  fill: var(--color-land);
}

.post-line {
  stroke: var(--color-faint);
  stroke-width: 6;
  stroke-linecap: round;
}

.mark {
  stroke: var(--color-ner);
  stroke-width: 4;
  stroke-linecap: round;
}

.arrow-line {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 7 6;
}

.arrow-head,
.pencil {
  fill: var(--color-ink);
}

.axis {
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.categories .bar {
  fill: var(--color-classification);
}

.entities .bar {
  fill: var(--color-ner);
}

.target {
  stroke: var(--color-ink);
  stroke-width: 2.5;
  stroke-dasharray: 7 5;
}

.target-text {
  font-size: var(--text-2xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.score {
  font-size: var(--text-xs);
  font-weight: 700;
}

.categories .score {
  fill: var(--color-classification);
}

.entities .score {
  fill: var(--color-ner);
}

.score.failed {
  fill: var(--color-negative);
}

.pass {
  fill: #15803d;
}

.fail {
  fill: var(--color-negative);
}

.tick {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
}
</style>
