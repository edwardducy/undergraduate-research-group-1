<script setup lang="ts">
import gsap from 'gsap'
import { MotionPathPlugin } from 'gsap/MotionPathPlugin'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import data from '../data/cyclones.json'
import { draw, enter, exit, moveLong, pop } from '../shared/motion'

gsap.registerPlugin(MotionPathPlugin)

const arm = 'M -7 0 C -7 -11 5 -17 17 -15 C 8 -12 3 -8 0 -7 Z'

// One year of the timeline lasts this many seconds on click 1.
const yearSeconds = 0.75
const cross = { duration: 0.45, ease: 'power1.inOut' }

const years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
const span = 884 / years.length
const dateX = (date: number) => date * span

const axisTop = 34
// The map window, with soft edges so the land and tracks fade out instead of stopping.
const frame = data.frame
const fadeWidth = 60
const fadeX = [frame.left, frame.left + fadeWidth * 1.5, frame.right - fadeWidth / 2, frame.right]
const fadeY = [frame.top, frame.top + fadeWidth / 3, frame.bottom - fadeWidth / 3, frame.bottom]
const fadeStops = (edges: number[]) => edges.map((edge, index) => ({
  offset: (edge - edges[0]) / (edges[3] - edges[0]),
  colour: index === 0 || index === 3 ? 'black' : 'white',
}))

// The bar chart of posts per cyclone.
const chartLeft = 48
const chartRight = 884
const base = 316
const perPost = 0.23
const postY = (posts: number) => base - posts * perPost
const gridValues = [0, 200, 400, 600, 800, 1000]
const limit = 400
const slot = (chartRight - chartLeft) / data.storms.length
const keptSlot = 62

let rank = 0
const cyclones = data.storms.map((storm, index) => {
  const kept = storm.posts >= limit
  return {
    ...storm,
    year: Math.floor(storm.date),
    kept,
    x: chartLeft + slot * (index + 0.5),
    keptX: kept ? chartLeft + keptSlot * (rank++ + 0.5) : 0,
    height: storm.posts * perPost,
  }
})
const narrowRight = chartLeft + keptSlot * rank + 24

// A bracket under the storms of each year.
const brackets = years.map((year, index) => {
  const slots = cyclones.flatMap((cyclone, slotIndex) => cyclone.year === index ? [slotIndex] : [])
  return {
    year,
    from: chartLeft + slot * slots[0] + 6,
    to: chartLeft + slot * (slots[slots.length - 1] + 1) - 6,
  }
})

// 100 blocks, each standing for about 39 posts.
const blockCount = 100
const small = { size: 24, step: 29, x: 599, y: 78 }
const large = { size: 30, step: 36, x: 265, y: 56 }
const onGrid = (grid: typeof small, index: number) => ({
  x: grid.x + (index % 10) * grid.step,
  y: grid.y + Math.floor(index / 10) * grid.step,
})

// One fixed random split into 72 training, 14 validation and 14 test blocks.
let seed = 7
const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647
const order = Array.from({ length: blockCount }, (_, index) => index)
for (let index = order.length - 1; index > 0; index--) {
  const other = Math.floor(random() * (index + 1))
  ;[order[index], order[other]] = [order[other], order[index]]
}
const groupNames = ['train', 'validation', 'test'] as const
const groupOf = Array.from<number>({ length: blockCount })
order.forEach((block, position) => (groupOf[block] = position < 72 ? 0 : position < 86 ? 1 : 2))

const groupBottom = 330
const groupGap = 80
const columns = [9, 2, 2]
const rows = [8, 7, 7]
const groupWidth = (group: number) => columns[group] * large.step - (large.step - large.size)
const groupX = [137]
groupX.push(groupX[0] + groupWidth(0) + groupGap)
groupX.push(groupX[1] + groupWidth(1) + groupGap)
const filled = [0, 0, 0]
const blocks = Array.from({ length: blockCount }, (_, index) => {
  const group = groupOf[index]
  const place = filled[group]++
  return {
    group,
    small: onGrid(small, index),
    large: onGrid(large, index),
    split: {
      x: groupX[group] + (place % columns[group]) * large.step,
      y: groupBottom - (rows[group] - Math.floor(place / columns[group])) * large.step + (large.step - large.size),
    },
  }
})
const groupLabels = [
  { name: 'Training', share: '72%', posts: 'about 2,800 posts' },
  { name: 'Validation', share: '14%', posts: 'about 550' },
  { name: 'Test', share: '14%', posts: 'about 550' },
].map((label, group) => ({ ...label, x: groupX[group] + groupWidth(group) / 2 }))

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()
  const shades = [colour('data'), colour('data-light'), colour('surface')]

  gsap.set('.track, .track-halo', { strokeDasharray: 1, strokeDashoffset: 1, opacity: 0 })
  gsap.set('.storm, .playhead, .dot, .big:not(.first), .chart, .bar-text, .brackets, .limit, .limit-text, .source, .grid-title, .grid-note, .pool-title, .pool-note, .legend, .group-label', { opacity: 0 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.base-axis', { attr: { x2: chartLeft } })
  gsap.set('.limit-line', { attr: { x2: chartLeft } })
  gsap.set('.block', { opacity: 0, scale: 0.4, transformOrigin: '50% 50%' })
  cyclones.forEach((cyclone, index) => gsap.set(`.storm-bar[data-index="${index}"]`, { x: dateX(cyclone.date), y: axisTop }))

  timeline.addLabel('step0')
  timeline.addLabel('run')
  const at = (date: number) => `run+=${date * yearSeconds}`
  const length = years.length * yearSeconds

  // The playhead sweeps the timeline at a steady pace and fills the axis behind it.
  timeline
    .set('.playhead', { opacity: 1 }, 'run')
    .to('.playhead', { x: dateX(years.length), duration: length, ease: 'none' }, 'run')
    .to('.axis-fill', { attr: { x2: dateX(years.length) }, duration: length, ease: 'none' }, 'run')

  for (let tick = 0; tick <= years.length; tick++)
    timeline.set(`.tick[data-index="${tick}"]`, { stroke: colour('data') }, at(tick))

  // Each new year takes focus in the labels and the large year.
  years.forEach((_, index) => {
    timeline
      .set(`.year-text[data-index="${index}"]`, { fill: colour('ink'), fontWeight: 700 }, at(index))
      .set(`.big[data-index="${index}"]`, { opacity: 1, fill: colour('ink') }, at(index))
    if (index > 0) {
      timeline
        .set(`.year-text[data-index="${index - 1}"]`, { fill: colour('muted'), fontWeight: 600 }, at(index))
        .set(`.big[data-index="${index - 1}"]`, { opacity: 0 }, at(index))
    }
  })

  // Each storm draws its track across the map as the playhead passes its date.
  cyclones.forEach((cyclone, index) => {
    const lines = `.track[data-index="${index}"], .track-halo[data-index="${index}"]`
    const storm = `.storm[data-index="${index}"]`
    timeline
      .set(lines, { opacity: 1 }, at(cyclone.date))
      .to(lines, { strokeDashoffset: 0, ...cross }, at(cyclone.date))
      .set(storm, { opacity: 1 }, at(cyclone.date))
      .to(storm, { motionPath: { path: cyclone.track }, ...cross }, at(cyclone.date))
      .fromTo(`${storm} .spin`, { rotation: 0 }, { rotation: -540, svgOrigin: '0 0', ...cross }, at(cyclone.date))
      .set(storm, { opacity: 0 }, `run+=${cyclone.date * yearSeconds + cross.duration}`)
      .to(`.storm-bar[data-index="${index}"] .dot`, { opacity: 1, ...pop }, at(cyclone.date))

    // A past year's track fades to grey once the year ends and its storm has crossed.
    const fade = Math.max(cyclone.year + 1, cyclone.date + cross.duration / yearSeconds)
    if (fade < years.length)
      timeline.to(`.track[data-index="${index}"]`, { stroke: colour('faint'), strokeWidth: 2.2, duration: 0.2 }, at(fade))
  })

  // At the end, the whole period shows and every track settles to the data colour.
  timeline
    .to('.playhead', { opacity: 0, ...exit }, at(years.length))
    .set(`.big[data-index="${years.length - 1}"]`, { opacity: 0 }, at(years.length))
    .set('.big.range', { opacity: 1 }, at(years.length))
    .set(`.year-text[data-index="${years.length - 1}"]`, { fill: colour('muted'), fontWeight: 600 }, at(years.length))
    .to('.track', { stroke: colour('data'), strokeWidth: 2.2, duration: 0.3 }, at(years.length))
    .addLabel('step1')

  // Click 2. The map leaves and each storm dot drops to its slot and grows a bar of its posts.
  timeline
    .to('.map, .tracks, .cyclone-label, .timeline', { opacity: 0, ...exit })
    .addLabel('drop')
    .to('.storm-bar', { x: (index: number) => cyclones[index].x, y: base, ...moveLong, stagger: 0.03 }, 'drop')
    .to('.bar', { scaleY: 1, duration: 0.8, ease: 'power2.out', stagger: 0.03 }, 'drop+=0.3')
    .to('.base-axis', { attr: { x2: chartRight }, ...draw }, 'drop+=0.2')
    .to('.chart, .source', { opacity: 1, ...enter }, 'drop+=0.4')
    .to('.bar-text, .brackets', { opacity: 1, ...enter }, 'drop+=0.9')
    .addLabel('step2')

  // Click 3. The 400 line draws, and the storms below it turn to outlines and sink into the axis.
    .addLabel('limit')
    .set('.limit', { opacity: 1 }, 'limit')
    .to('.limit-line', { attr: { x2: chartRight }, duration: 0.6, ease: 'power1.inOut' }, 'limit')
    .to('.dropped .bar', { fill: colour('land'), strokeOpacity: 1, ...enter }, 'limit+=0.3')
    .to('.dropped .value', { fill: colour('faint'), ...enter }, 'limit+=0.3')
    .to('.dropped .bar', { scaleY: 0, duration: 0.5, ease: 'power2.in' }, 'limit+=1')
    .to('.dropped .value', { opacity: 0, ...exit }, 'limit+=1')
    .to('.dropped .dot', { stroke: colour('faint'), ...enter }, 'limit+=1.1')
    .to('.limit-text', { opacity: 1, ...enter }, 'limit+=1.1')
    .addLabel('step3')

  // Click 4. The kept bars close the gaps and the posts appear as 100 blocks.
    .addLabel('gather')
    .to('.dropped, .brackets', { opacity: 0, ...exit }, 'gather')
    .to('.kept', { x: (_: number, element: Element) => Number(element.getAttribute('data-kept-x')), ...moveLong }, 'gather+=0.1')
    .to('.grid-line, .base-axis, .limit-line', { attr: { x2: narrowRight }, ...moveLong }, 'gather+=0.1')
    .to('.limit-text', { attr: { x: narrowRight + 8 }, ...moveLong }, 'gather+=0.1')
    .to('.grid-title', { opacity: 1, ...enter }, 'gather+=0.6')
    .to('.block', { opacity: 1, scale: 1, ...pop, stagger: (index: number) => Math.floor(index / 10) * 0.08 }, 'gather+=0.7')
    .to('.grid-note', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step4')

  // Click 5. The chart leaves and the grid of posts moves to the centre and grows.
    .addLabel('pool')
    .to('.chart, .base-axis, .storm-bar, .limit, .source, .grid-title, .grid-note', { opacity: 0, ...exit }, 'pool')
    .to('.block', { attr: { x: (index: number) => blocks[index].large.x, y: (index: number) => blocks[index].large.y, width: large.size, height: large.size }, ...moveLong }, 'pool+=0.1')
    .to('.pool-title, .pool-note', { opacity: 1, ...enter }, 'pool+=0.6')
    .addLabel('step5')

  // Click 6. Every block takes one of three shades at once, which shows a single random split.
    .addLabel('tint')
    .to('.pool-note', { opacity: 0, ...exit }, 'tint')
    .to('.block', { fill: (index: number) => shades[blocks[index].group], stroke: colour('data'), strokeWidth: (index: number) => blocks[index].group === 2 ? 2 : 0, ...enter }, 'tint')
    .to('.legend', { opacity: 1, ...enter }, 'tint+=0.1')
    .addLabel('step6')

  // Click 7. The blocks slide into the training, validation and test groups.
    .addLabel('split')
    .to('.pool-title, .legend', { opacity: 0, ...exit }, 'split')
    .to('.block', { attr: { x: (index: number) => blocks[index].split.x, y: (index: number) => blocks[index].split.y }, ...moveLong, stagger: { amount: 0.3, from: 'random' } }, 'split')
    .to('.group-label', { opacity: 1, ...enter, stagger: 0.1 }, '>-0.2')
    .addLabel('step7')
})
</script>

<template>
  <div ref="root" class="dataset">
    <svg viewBox="0 0 884 384" width="884" height="384">
      <g class="cyclone-label">
        <text class="big first" data-index="0" x="0" y="190">2020</text>
        <text v-for="(year, index) in years.slice(1)" :key="year" class="big" :data-index="index + 1" x="0" y="190">{{ year }}</text>
        <text class="big range" x="0" y="190">2020–2026</text>
        <text class="caption" x="0" y="239">Tropical cyclones that</text>
        <text class="caption" x="0" y="271">crossed the Philippines</text>
      </g>

      <defs>
        <linearGradient id="dataset-fade-x" gradientUnits="userSpaceOnUse" :x1="fadeX[0]" :x2="fadeX[3]" y1="0" y2="0">
          <stop v-for="stop in fadeStops(fadeX)" :key="stop.offset" :offset="stop.offset" :stop-color="stop.colour" />
        </linearGradient>
        <linearGradient id="dataset-fade-y" gradientUnits="userSpaceOnUse" x1="0" x2="0" :y1="fadeY[0]" :y2="fadeY[3]">
          <stop v-for="stop in fadeStops(fadeY)" :key="stop.offset" :offset="stop.offset" :stop-color="stop.colour" />
        </linearGradient>
        <mask id="dataset-mask-x" maskUnits="userSpaceOnUse" :x="frame.left" :y="frame.top" :width="frame.right - frame.left" :height="frame.bottom - frame.top">
          <rect :x="frame.left" :y="frame.top" :width="frame.right - frame.left" :height="frame.bottom - frame.top" fill="url(#dataset-fade-x)" />
        </mask>
        <mask id="dataset-mask-y" maskUnits="userSpaceOnUse" :x="frame.left" :y="frame.top" :width="frame.right - frame.left" :height="frame.bottom - frame.top">
          <rect :x="frame.left" :y="frame.top" :width="frame.right - frame.left" :height="frame.bottom - frame.top" fill="url(#dataset-fade-y)" />
        </mask>
      </defs>

      <g mask="url(#dataset-mask-x)">
        <g mask="url(#dataset-mask-y)">
          <path class="map" :d="data.land" />
          <g class="tracks">
            <path v-for="(cyclone, index) in cyclones" :key="`halo-${index}`" class="track-halo" :data-index="index" :d="cyclone.track" pathLength="1" />
            <path v-for="(cyclone, index) in cyclones" :key="index" class="track" :data-index="index" :d="cyclone.track" pathLength="1" />
          </g>
          <g v-for="(cyclone, index) in cyclones" :key="`storm-${index}`" class="storm" :data-index="index">
            <g class="spin">
              <g class="halo" transform="scale(1.4)">
                <circle r="9" />
                <path :d="arm" />
                <path :d="arm" transform="rotate(180)" />
              </g>
              <g class="symbol" transform="scale(1.4)">
                <path :d="arm" />
                <path :d="arm" transform="rotate(180)" />
                <circle r="7" />
                <circle class="eye" r="2.5" />
              </g>
            </g>
          </g>
        </g>
      </g>

      <g class="chart">
        <text class="chart-label" :x="chartLeft" y="44">Taglish posts after cleaning</text>
        <g v-for="value in gridValues" :key="value">
          <text class="axis-value" :x="chartLeft - 10" :y="postY(value) + 5" text-anchor="end">{{ value.toLocaleString('en-US') }}</text>
          <line v-if="value > 0" class="grid-line" :x1="chartLeft" :y1="postY(value)" :x2="chartRight" :y2="postY(value)" />
        </g>
      </g>

      <g class="timeline">
        <path class="axis" :d="`M 0 ${axisTop} H 884`" />
        <line class="axis-fill" x1="0" :y1="axisTop" x2="0" :y2="axisTop" />
        <path v-for="tick in years.length + 1" :key="`tick-${tick}`" class="tick" :data-index="tick - 1" :d="`M ${dateX(tick - 1)} ${axisTop - 8} V ${axisTop + 8}`" />
        <text v-for="(year, index) in years" :key="year" class="year-text" :data-index="index" :x="dateX(index + 0.5)" :y="axisTop - 22" text-anchor="middle">{{ year }}</text>
        <circle class="playhead" cx="0" :cy="axisTop" r="9" />
      </g>

      <line class="base-axis" :x1="chartLeft" :y1="base" :x2="chartRight" :y2="base" />
      <g class="brackets">
        <g v-for="bracket in brackets" :key="bracket.year">
          <line class="bracket" :x1="bracket.from" :y1="base + 24" :x2="bracket.to" :y2="base + 24" />
          <text class="bracket-year" :x="(bracket.from + bracket.to) / 2" :y="base + 44" text-anchor="middle">{{ bracket.year }}</text>
        </g>
      </g>

      <g
        v-for="(cyclone, index) in cyclones" :key="`bar-${index}`" class="storm-bar" :class="cyclone.kept ? 'kept' : 'dropped'"
        :data-index="index" :data-kept-x="cyclone.keptX"
      >
        <rect class="bar" x="-17" :y="-cyclone.height" width="34" :height="cyclone.height" />
        <text class="bar-text value" x="0" :y="-cyclone.height - 8" text-anchor="middle">{{ cyclone.posts }}</text>
        <circle class="dot" cx="0" cy="0" r="6" />
      </g>

      <g class="limit">
        <line class="limit-line" :x1="chartLeft" :y1="postY(limit)" :x2="chartRight" :y2="postY(limit)" />
        <text class="limit-text" :x="chartRight + 8" :y="postY(limit) + 6">400</text>
      </g>

      <text class="grid-title" :x="small.x" y="56"><tspan class="grid-word">About </tspan><tspan class="grid-number">3,900</tspan><tspan class="grid-word"> posts</tspan></text>
      <text class="note grid-note" :x="small.x" :y="small.y + 10 * small.step + 16">1 block = about 39 posts</text>

      <text class="pool-title" x="442" y="30" text-anchor="middle">About 3,900 labeled posts</text>
      <text class="note pool-note" x="442" :y="large.y + 10 * large.step + 22" text-anchor="middle">1 block = about 39 posts</text>

      <rect
        v-for="(block, index) in blocks" :key="`block-${index}`" class="block"
        :x="block.small.x" :y="block.small.y" :width="small.size" :height="small.size" rx="3"
      />

      <g class="legend">
        <g v-for="(name, group) in groupNames" :key="name" :transform="`translate(${282 + group * 120} 432)`">
          <rect class="swatch" :class="name" x="0" y="-14" width="16" height="16" rx="3" />
          <text class="legend-text" x="24" y="0">{{ groupLabels[group].name }}</text>
        </g>
      </g>

      <g v-for="label in groupLabels" :key="label.name" class="group-label">
        <text class="name" :x="label.x" :y="groupBottom + 34" text-anchor="middle">{{ label.name }}</text>
        <text class="share" :x="label.x" :y="groupBottom + 60" text-anchor="middle">{{ label.share }}</text>
        <text class="note" :x="label.x" :y="groupBottom + 82" text-anchor="middle">{{ label.posts }}</text>
      </g>
    </svg>

    <footer class="source">Source: Seeberger et al. (2025)</footer>
  </div>
</template>

<style scoped>
svg {
  overflow: visible;
}

.dataset {
  position: relative;
  height: 456px;
}

footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}

.map {
  fill: var(--color-land);
}

.big {
  font-size: calc(var(--text-display) * 1.3);
  font-weight: 700;
  letter-spacing: -0.01em;
  fill: var(--color-ink);
}

.big.first {
  fill: var(--color-faint);
}

.big.range {
  font-size: var(--text-display);
}

.caption {
  font-size: var(--text-md);
  font-weight: 600;
  fill: var(--color-ink);
}

.track {
  fill: none;
  stroke: var(--color-data);
  stroke-width: 3;
  stroke-linecap: round;
}

.track-halo {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 7;
  stroke-linecap: round;
}

.halo {
  fill: var(--color-surface);
  stroke: var(--color-surface);
  stroke-width: 5;
  stroke-linejoin: round;
}

.symbol {
  fill: var(--color-data);
}

.eye {
  fill: var(--color-surface);
}

.axis,
.tick {
  stroke: var(--color-faint);
  stroke-width: 2;
}

.axis {
  stroke-width: 4;
  stroke-linecap: round;
}

.axis-fill {
  stroke: var(--color-data);
  stroke-width: 4;
  stroke-linecap: round;
}

.year-text {
  font-size: var(--text-xs);
  font-weight: 600;
  fill: var(--color-faint);
}

.playhead {
  fill: var(--color-data);
}

.dot {
  fill: var(--color-surface);
  stroke: var(--color-data);
  stroke-width: 2.5;
}

.chart-label {
  font-size: var(--text-base);
  font-weight: 600;
  fill: var(--color-muted);
}

.axis-value {
  font-size: 0.95rem;
  fill: var(--color-subtle);
}

.grid-line {
  stroke: var(--color-land);
  stroke-width: 1.5;
}

.base-axis {
  stroke: var(--color-ink);
  stroke-width: 3;
}

.bracket {
  stroke: var(--color-faint);
  stroke-width: 1.5;
}

.bracket-year {
  font-size: 0.95rem;
  font-weight: 600;
  fill: var(--color-subtle);
}

.bar {
  fill: var(--color-data);
  stroke: var(--color-subtle);
  stroke-opacity: 0;
  stroke-width: 2;
  stroke-dasharray: 6 4;
}

.value {
  font-size: 0.875rem;
  fill: var(--color-subtle);
}

.limit-line {
  stroke: var(--color-subtle);
  stroke-width: 2.5;
  stroke-dasharray: 10 7;
}

.limit-text {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.grid-number {
  font-size: var(--text-4xl);
  font-weight: 700;
  letter-spacing: -0.02em;
  fill: var(--color-ink);
}

.grid-word {
  font-size: var(--text-base);
  font-weight: 600;
  fill: var(--color-muted);
}

.note {
  font-size: 0.95rem;
  fill: var(--color-subtle);
}

.pool-title {
  font-size: var(--text-xl);
  font-weight: 700;
  fill: var(--color-ink);
}

.block {
  fill: var(--color-data);
  stroke: var(--color-data);
  stroke-width: 0;
}

.swatch.train {
  fill: var(--color-data);
}

.swatch.validation {
  fill: var(--color-data-light);
}

.swatch.test {
  fill: var(--color-surface);
  stroke: var(--color-data);
  stroke-width: 2;
}

.legend-text {
  font-size: 0.95rem;
  fill: var(--color-muted);
}

.group-label .name {
  font-size: var(--text-base);
  font-weight: 700;
  fill: var(--color-ink);
}

.share {
  font-size: var(--text-sm);
  font-weight: 600;
  fill: var(--color-muted);
}
</style>
