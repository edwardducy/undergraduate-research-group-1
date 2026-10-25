<script setup lang="ts">
import gsap from 'gsap'
import { MotionPathPlugin } from 'gsap/MotionPathPlugin'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
// Built by scripts/luzon_map.py. The posts are an illustration placed by population, not real posts
import map from '../data/luzon-map.json'

gsap.registerPlugin(MotionPathPlugin)

const posts = map.posts.map(([x, y, request]) => ({ x, y, request: request === 1 }))
const requests = posts.filter(post => post.request)

// Northern hemisphere tropical cyclone symbol: a solid centre with two arms trailing clockwise
const arm = 'M -7 0 C -7 -11 5 -17 17 -15 C 8 -12 3 -8 0 -7 Z'

// The same motion tokens as slide 2, plus a slower one for the storm crossing the whole map
// and a quick overshoot for each post popping up
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const travel = { duration: 1.2, ease: 'power1.inOut' }
const pop = { duration: 0.25, ease: 'back.out(3)' }
const sweep = { duration: 0.35, ease: 'power2.inOut' }
const moveShort = { duration: 0.5, ease: 'power2.inOut' }
const dial = { duration: 0.45, ease: 'power2.inOut' }
// How far a label travels as the dial turns by one row, and how flat it gets at the wheel's edge
const roll = 16
const edge = 0.2

const root = ref<HTMLElement>()

const figures = [
  { key: 'rank', number: '1st', lines: ['Highest disaster risk', 'among 193 countries'] },
  { key: 'cyclones', number: '8–9', lines: ['Tropical cyclones cross', 'the country each year'] },
]

// A block in the text's colour sweeps in over the number, the number appears under it,
// and the block sweeps out the other side. The statement then fades in like the rest of the deck
function revealFigure(timeline: gsap.core.Timeline, figure: string) {
  timeline
    .to(`${figure} .block`, { scaleX: 1, ...sweep })
    .set(`${figure} .text`, { opacity: 1 })
    .set(`${figure} .block`, { transformOrigin: '100% 50%' })
    .to(`${figure} .block`, { scaleX: 0, ...sweep })
    .to(`${figure} .statement`, { opacity: 1, ...enter }, '<')
}

useClickTimeline(root, (timeline) => {
  gsap.set('.storm, .track, .casing, .requests .post, .legend .one, .legend .request-key, .legend .other-key > *, .figure .text, .statement', { opacity: 0 })
  gsap.set('.legend .one text, .legend .other-key text', { transformOrigin: '50% 50%' })
  gsap.set('.figure .block', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.track, .casing', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.posts .post', { scale: 0, transformOrigin: '50% 50%' })
  gsap.set('.requests', { opacity: 0 })

  timeline.addLabel('step0')
  revealFigure(timeline, '.rank')
  timeline.addLabel('step1')
  revealFigure(timeline, '.cyclones')
  timeline
    .addLabel('step2')
    .to('.storm, .track, .casing', { opacity: 1, ...enter })
    .to('.track, .casing', { strokeDashoffset: 0, ...travel }, '<')
    .to('.storm', { motionPath: { path: map.track }, ...travel }, '<')
    .addLabel('step3')
    // A few posts pop up at first, then they arrive faster and faster until they flood the map
    .to('.posts .post', { scale: 1, ...pop, stagger: { amount: 2, from: 'random', ease: 'power3.out' } })
    .to('.legend .one', { opacity: 1, ...enter }, '<')
    .addLabel('step4')
    // "One post" first slides down to the row of "Other posts". After a short pause, a dial turns
    // its words into "Other posts". Like a picker wheel, the old words roll up and away while the
    // new ones roll up from below, and squashing them as they move stands in for the wheel's curve.
    // The dot stays in place and only fades to the lighter grey of the other posts
    .to('.legend .one', { y: 28, ...moveShort })
    .to('.legend .one text', { y: -roll, scaleY: edge, opacity: 0, ...dial }, '+=0.25')
    .fromTo('.legend .other-key text', { y: roll, scaleY: edge, opacity: 0 }, { y: 0, scaleY: 1, opacity: 1, ...dial }, '<')
    .to('.legend .one circle', { opacity: 0, ...dial }, '<')
    .to('.legend .other-key circle', { opacity: 1, ...dial }, '<')
    .to('.posts', { opacity: 0.35, ...exit }, '<')
    .to('.requests, .requests .post', { opacity: 1, ...enter }, '<')
    .to('.legend .request-key', { opacity: 1, ...enter }, '<')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="risk-map">
    <div class="figures">
      <div v-for="figure in figures" :key="figure.key" class="figure" :class="figure.key">
        <div class="number"><span class="reveal"><span class="text">{{ figure.number }}</span><span class="block" /></span></div>
        <div class="statement">
          <template v-for="(line, index) in figure.lines" :key="index">
            <br v-if="index > 0">{{ line }}
          </template>
        </div>
      </div>
    </div>

    <svg viewBox="0 0 588 504" width="588" height="504">
      <path class="land" :d="map.land" />

      <g class="posts">
        <circle v-for="(post, index) in posts" :key="index" class="post" :cx="post.x" :cy="post.y" r="2.2" />
      </g>
      <g class="requests">
        <circle v-for="(post, index) in requests" :key="index" class="post request" :cx="post.x" :cy="post.y" r="2.2" />
      </g>

      <path class="casing" :d="map.track" pathLength="1" />
      <path class="track" :d="map.track" pathLength="1" />
      <g class="storm">
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

      <g class="legend">
        <g class="one">
          <circle class="post" cx="300" cy="12" r="5" />
          <text x="312" y="18">One post</text>
        </g>
        <g class="sorted">
          <g class="request-key">
            <circle class="post request" cx="300" cy="12" r="5" />
            <text x="312" y="18">Requests or urgent needs</text>
          </g>
          <g class="other-key">
            <circle class="post faded" cx="300" cy="40" r="5" />
            <text x="312" y="46">Other posts</text>
          </g>
        </g>
      </g>
    </svg>

    <footer>Sources: WorldRiskReport 2025, PAGASA</footer>
  </div>
</template>

<style scoped>
.risk-map {
  position: relative;
  display: grid;
  grid-template-columns: 1fr 540px;
  column-gap: var(--space-4);
  height: 456px;
}

/* The map runs into the slide's right and bottom margins, so the land ends at the slide's edge */
svg {
  width: calc(100% + var(--space-5));
  height: calc(100% + var(--space-5));
}

.figures {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  padding-top: var(--space-3);
}

.number {
  font-size: var(--text-display);
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}

.statement {
  margin-top: var(--space-1);
  font-size: var(--text-md);
  font-weight: 600;
  line-height: 1.3;
}

.reveal {
  position: relative;
  display: inline-block;
}

/* The block takes the colour of the text it reveals. Every block starts exactly at the text's
   left edge, so the blocks of all lines line up on the left */
.block {
  position: absolute;
  top: -0.08em;
  bottom: -0.08em;
  left: 0;
  right: -0.08em;
  background: currentColor;
}

.land {
  fill: var(--color-land);
}

.post {
  fill: var(--color-ink);
  fill-opacity: 0.3;
}

.post.request {
  fill: var(--color-classification);
  fill-opacity: 0.6;
}

.post.faded {
  fill-opacity: 0.1;
}

.track,
.casing {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.track {
  stroke: var(--color-ink);
  stroke-width: 2.5;
}

.casing,
.halo {
  stroke: var(--color-surface);
  stroke-width: 8;
}

.halo {
  fill: var(--color-surface);
  stroke-width: 5;
  stroke-linejoin: round;
}

.symbol {
  fill: var(--color-ink);
}

.eye {
  fill: var(--color-surface);
}

.legend text {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
}

/* The sources sit below the map, in the slide's bottom margin */
footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}

</style>
