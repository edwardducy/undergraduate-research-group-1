<script setup lang="ts">
import gsap from 'gsap'
import { MotionPathPlugin } from 'gsap/MotionPathPlugin'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
// The same map as the disaster risk slide
import map from '../data/luzon-map.json'

gsap.registerPlugin(MotionPathPlugin)

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }
const pop = { duration: 0.25, ease: 'back.out(3)' }
// One cyclone crossing the map. Shorter than the disaster risk slide's, since 14 cross in a row
const cross = { duration: 0.45, ease: 'power1.inOut' }

const dim = 0.35

// The same tropical cyclone symbol as the disaster risk map
const arm = 'M -7 0 C -7 -11 5 -17 17 -15 C 8 -12 3 -8 0 -7 Z'

// Scene 1. The timeline across the top, the map below it. The map is drawn large enough to run
// past the drawing area and off the bottom of the slide, so the islands are cut by the slide's
// edge instead of by the margin
const axisTop = 52
const mapScale = 0.75
const mapX = (884 - 588 * mapScale) / 2
const mapY = 96
// Scene 2. The same timeline, dropped to the foot of a bar chart of posts per cyclone
const axisBottom = 312
const perPost = 0.37
const limitY = axisBottom - 400 * perPost

const years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
const yearX = (index: number) => 112 + index * 110

// An illustration. The real cyclones and counts are not known until the posts are collected.
// The counts are chosen so the kept cyclones add up to 3,900
const counts: [number, number][] = [
  [0, 430], [0, 180], [1, 520], [1, 320], [2, 90], [2, 610], [3, 450],
  [3, 260], [4, 470], [4, 350], [5, 560], [5, 140], [6, 410], [6, 450],
]

let rank = 0
const keptCount = counts.filter(([, posts]) => posts >= 400).length
const cyclones = counts.map(([year, posts], index) => {
  const x = yearX(year) + (index % 2 === 0 ? -11 : 11)
  const kept = posts >= 400
  // Each track enters from the Pacific on the right and heads west-northwest across the islands
  // Each stays between the timeline and the footer, so the storm never covers either
  const leaveY = 110 + (index * 37) % 130
  const enterY = Math.min(330, leaveY + 60 + (index * 23) % 110)
  const bend = (index * 19) % 40 - 20
  const y = (at: number) => leaveY + (enterY - leaveY) * (at + 20) / 920
  const track = `M 900 ${enterY} C 620 ${y(620) + bend} 300 ${y(300) - bend} -20 ${leaveY}`
  // The kept cyclones close up into a row on the left once the others are gone
  const shift = kept ? 60 + (rank++) * 26 - x : 0
  return { year, posts, x, kept, track, shift, height: posts * perPost }
})

// Each square stands for about 78 of the 3,900 posts, so 50 squares in all.
// The groups follow the 72, 14, and 14 percent split of Chapter 3
const size = 22
const step = 30
const squares = Array.from({ length: 50 }, (_, index) => ({
  group: index < 36 ? 'train' : index < 43 ? 'validation' : 'test',
  // Its place within its own part
  slot: index < 36 ? index : index < 43 ? index - 36 : index - 43,
  x: 296 + (index % 10) * step,
  y: 90 + Math.floor(index / 10) * step,
}))

function place(origin: { x: number, y: number }, columns: number) {
  return (index: number) => ({
    x: origin.x + (index % columns) * step,
    y: origin.y + Math.floor(index / columns) * step,
  })
}

const groups = {
  train: { place: place({ x: 60, y: 92 }, 9), centre: 191 },
  validation: { place: place({ x: 414, y: 92 }, 4), centre: 470 },
  test: { place: place({ x: 684, y: 92 }, 4), centre: 740 },
}

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const token = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  // The pool is one shade until each part takes its own shade as it moves out
  const shades = { train: token('--color-subtle'), validation: token('--color-ink'), test: token('--color-surface') }

  gsap.set('.track', { strokeDasharray: 1, strokeDashoffset: 1, opacity: 0 })
  gsap.set('.storm, .playhead, .dot, .source, .chart-label, .limit, .total, .pool-label, .square, .group-label', { opacity: 0 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })

  timeline.addLabel('step0')

  // Sentence 1. Year by year, each cyclone crosses the map, then lands on its year
  cyclones.forEach((cyclone, index) => {
    const track = `.track[data-index="${index}"]`
    timeline
      .to('.playhead', { opacity: 1, x: yearX(cyclone.year), duration: index === 0 ? 0 : 0.2, ease: 'power1.inOut' }, index === 0 ? '<' : '>')
      .set('.storm', { opacity: 1 }, '<')
      .set(track, { opacity: 1 }, '<')
      .to(track, { strokeDashoffset: 0, ...cross }, '<')
      .to('.storm', { motionPath: { path: cyclone.track }, ...cross }, '<')
      // Counterclockwise, as typhoons turn north of the equator
      .fromTo('.spin', { rotation: 0 }, { rotation: -540, svgOrigin: '0 0', ...cross }, '<')
      .to(track, { stroke: token('--color-subtle'), strokeWidth: 2, opacity: 0.6, duration: 0.2 })
      .to(`.dot[data-index="${index}"]`, { opacity: 1, ...pop }, '<')
  })
  timeline
    .to('.storm, .playhead', { opacity: 0, ...exit })
    .addLabel('step1')

  // Sentence 2 begins. The map fades and the timeline drops to become the foot of a chart.
  // Each cyclone rises into its number of posts
  timeline
    .to('.map, .tracks, .cyclone-label', { opacity: 0, ...exit })
    .to('.timeline', { y: axisBottom - axisTop, ...moveLong })
    .to('.chart-label, .source', { opacity: 1, ...enter }, '<0.3')
    .to('.dot', { opacity: 0, ...exit }, '>-0.1')
    .to('.bar', { scaleY: 1, ...moveLong, stagger: 0.03 }, '<')
    .addLabel('step2')

  // Only cyclones with at least 400 Taglish posts are kept. The others fade, then disappear
    .to('.limit', { opacity: 1, ...enter })
    .to('.bar.dropped', { opacity: dim, ...exit }, '>0.1')
    .to('.bar.dropped', { opacity: 0, duration: 0.6, ease: 'power1.in' }, '+=0.5')
    .addLabel('step3')

  // The kept cyclones close up and give about 3,900 posts
    .to('.limit, .year', { opacity: 0, ...exit })
    .to('.bar.kept', { x: (_: number, element: Element) => Number(element.getAttribute('data-shift')), ...moveLong }, '<')
    .to('.total', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step4')

  // The posts, about 3,900 of them
    .to('.timeline, .chart-label, .total, .source', { opacity: 0, ...exit })
    .to('.pool-label', { opacity: 1, ...enter })
    .to('.square', { opacity: 1, ...pop, stagger: { amount: 0.5, from: 'random' } }, '<')
    .addLabel('step5')

  // The pool splits into its three parts at once, then each part takes its shade and job
  // on its own click, as the script names it
  const target = (element: Element) => {
    const group = groups[element.getAttribute('data-group') as keyof typeof groups]
    return group.place(Number(element.getAttribute('data-slot')))
  }
  timeline
    .to('.pool-label', { opacity: 0, ...exit })
    .to('.square', { attr: { x: (_: number, element: Element) => target(element).x, y: (_: number, element: Element) => target(element).y }, ...moveLong }, '<')

  let count = 5
  for (const name of Object.keys(groups)) {
    timeline
      .to(`.square.${name}`, { fill: shades[name as keyof typeof shades], strokeWidth: name === 'test' ? 2 : 0, ...enter }, count === 5 ? '>-0.2' : '>')
      .to(`.group-label.${name}`, { opacity: 1, ...enter }, '<')
      .addLabel(`step${++count}`)
  }
})
</script>

<template>
  <div ref="root" class="dataset">
    <svg viewBox="0 0 884 384" width="884" height="384">
      <defs>
        <!-- The tracks stay below the timeline and inside the slide, down to its bottom edge -->
        <clipPath id="dataset-below">
          <rect x="0" y="92" width="884" height="412" />
        </clipPath>
      </defs>

      <!-- Scene 1. The cyclones that crossed the Philippines from 2020 to 2026 -->
      <text class="tag cyclone-label" x="442" y="14" text-anchor="middle">Tropical cyclones that crossed the Philippines</text>
      <path class="map" :d="map.land" :transform="`translate(${mapX} ${mapY}) scale(${mapScale})`" />
      <g class="tracks" clip-path="url(#dataset-below)">
        <path v-for="(cyclone, index) in cyclones" :key="index" class="track" :data-index="index" :d="cyclone.track" pathLength="1" />
      </g>
      <g class="storm">
        <g class="spin">
          <g class="halo" transform="scale(1.3)">
            <circle r="9" />
            <path :d="arm" />
            <path :d="arm" transform="rotate(180)" />
          </g>
          <g class="symbol" transform="scale(1.3)">
            <path :d="arm" />
            <path :d="arm" transform="rotate(180)" />
            <circle r="7" />
            <circle class="eye" r="2.5" />
          </g>
        </g>
      </g>

      <!-- Scene 2. Taglish posts per cyclone, and the 400-post line -->
      <text class="tag chart-label" x="0" y="14">Taglish posts after cleaning, per cyclone</text>
      <g class="limit">
        <path class="limit-line" :d="`M 20 ${limitY} H 864`" />
        <text class="limit-text" x="884" :y="limitY - 8" text-anchor="end">At least</text>
        <text class="limit-text" x="884" :y="limitY + 22" text-anchor="end">400 posts</text>
      </g>

      <!-- The timeline is shared by both scenes. It moves down as a whole between them -->
      <g class="timeline">
        <path class="axis" :d="`M 20 ${axisTop} H 864`" />
        <g v-for="(year, index) in years" :key="year" class="year">
          <path class="tick" :d="`M ${yearX(index)} ${axisTop - 6} V ${axisTop + 6}`" />
          <text class="year-text" :x="yearX(index)" :y="axisTop + 30" text-anchor="middle">{{ year }}</text>
        </g>
        <rect
          v-for="(cyclone, index) in cyclones" :key="`bar-${index}`" class="bar" :class="cyclone.kept ? 'kept' : 'dropped'"
          :data-shift="cyclone.shift" :x="cyclone.x - 8" :y="axisTop - cyclone.height" width="16" :height="cyclone.height" rx="2"
        />
        <circle v-for="(cyclone, index) in cyclones" :key="`dot-${index}`" class="dot" :data-index="index" :cx="cyclone.x" :cy="axisTop" r="7" />
        <path class="playhead" :d="`M 0 ${axisTop - 16} V ${axisTop + 14}`" />
      </g>

      <g class="total">
        <text class="total-number" x="600" y="210" text-anchor="middle">About 3,900 posts</text>
        <text class="total-from" x="600" y="244" text-anchor="middle">From {{ keptCount }} cyclones</text>
      </g>

      <!-- The posts, then their three parts -->
      <text class="tag pool-label" x="442" y="62" text-anchor="middle">About 3,900 labeled Taglish posts</text>

      <rect v-for="(square, index) in squares" :key="index" class="square" :class="square.group" :data-group="square.group" :data-slot="square.slot" :x="square.x" :y="square.y" :width="size" :height="size" rx="4" />

      <g class="group-label train">
        <!-- An open book -->
        <g :transform="`translate(${groups.train.centre} 252)`">
          <path class="icon" d="M 0 -10 C -6 -14 -14 -14 -18 -12 V 10 C -14 8 -6 8 0 12 C 6 8 14 8 18 10 V -12 C 14 -14 6 -14 0 -10 Z M 0 -10 V 12" />
        </g>
        <text class="name" :x="groups.train.centre" y="300" text-anchor="middle">Training</text>
        <text class="job" :x="groups.train.centre" y="326" text-anchor="middle">Models learn here</text>
      </g>

      <g class="group-label validation">
        <!-- Three sliders -->
        <g :transform="`translate(${groups.validation.centre} 252)`">
          <path class="icon" d="M -16 -9 H 16 M -16 0 H 16 M -16 9 H 16" />
          <circle class="knob" cx="-6" cy="-9" r="3.5" />
          <circle class="knob" cx="8" cy="0" r="3.5" />
          <circle class="knob" cx="-2" cy="9" r="3.5" />
        </g>
        <text class="name" :x="groups.validation.centre" y="300" text-anchor="middle">Validation</text>
        <text class="job" :x="groups.validation.centre" y="326" text-anchor="middle">Every choice, such as</text>
        <text class="job" :x="groups.validation.centre" y="348" text-anchor="middle">the optimizer</text>
      </g>

      <g class="group-label test">
        <!-- A lock -->
        <g :transform="`translate(${groups.test.centre} 252)`">
          <path class="icon" d="M -8 -2 V -8 A 8 8 0 0 1 8 -8 V -2" />
          <rect class="icon" x="-13" y="-2" width="26" height="18" rx="3" />
        </g>
        <text class="name" :x="groups.test.centre" y="300" text-anchor="middle">Test</text>
        <text class="job" :x="groups.test.centre" y="326" text-anchor="middle">Used once,</text>
        <text class="job" :x="groups.test.centre" y="348" text-anchor="middle">at the very end</text>
      </g>
    </svg>

    <!-- In the same place as the sources on the earlier slides. Shown only with the chart -->
    <footer class="source">Source: Seeberger et al. (2025)</footer>
  </div>
</template>

<style scoped>
/* Lets the map run into the slide's bottom margin. The slide itself cuts it at its edge */
svg {
  overflow: visible;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.job {
  font-size: var(--text-xs);
  fill: var(--color-ink);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
  fill: var(--color-ink);
}

/* The full height of a slide with no heading, so the source sits where it does on slides 3 and 4 */
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

.track {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2.5;
  stroke-linecap: round;
}

.halo {
  fill: var(--color-surface);
  stroke: var(--color-surface);
  stroke-width: 5;
  stroke-linejoin: round;
}

.symbol {
  fill: var(--color-ink);
}

.eye {
  fill: var(--color-surface);
}

.axis,
.tick {
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.year-text {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
}

.playhead {
  stroke: var(--color-ink);
  stroke-width: 3;
}

.dot,
.bar {
  fill: var(--color-ink);
}

.limit-line {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 7 6;
}

.limit-text {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.total-number {
  font-size: var(--text-4xl);
  font-weight: 700;
  letter-spacing: -0.02em;
  fill: var(--color-ink);
}

.total-from {
  font-size: var(--text-xs);
  fill: var(--color-muted);
}

/* Neutral shades, since blue and amber already stand for the two tasks.
   The test posts end up as outlines, set aside and untouched until the end */
.square {
  fill: var(--color-subtle);
  stroke: var(--color-ink);
  stroke-width: 0;
}

.icon {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.knob {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}
</style>
