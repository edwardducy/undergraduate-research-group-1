<script setup lang="ts">
import gsap from 'gsap'
import { MotionPathPlugin } from 'gsap/MotionPathPlugin'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import map from '../data/luzon-map.json'
import { enter, exit, pop } from '../shared/motion'

gsap.registerPlugin(MotionPathPlugin)

const posts = map.posts.map(([x, y, request]) => ({ x, y, request: request === 1 }))
const requests = posts.filter(post => post.request)

const arm = 'M -7 0 C -7 -11 5 -17 17 -15 C 8 -12 3 -8 0 -7 Z'

const travel = { duration: 1.2, ease: 'power1.inOut' }

const flood = (progress: number) => progress ** (1 / 3)

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.rank, .cyclones, .storm, .track, .casing, .requests .post, .legend > *', { opacity: 0 })
  gsap.set('.track, .casing', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.posts .post', { scale: 0, transformOrigin: '50% 50%' })
  gsap.set('.requests', { opacity: 0 })

  timeline
    .addLabel('step0')
    .to('.rank', { opacity: 1, ...enter })
    .addLabel('step1')
    .to('.cyclones', { opacity: 1, ...enter })
    .addLabel('step2')
    .to('.storm, .track, .casing', { opacity: 1, ...enter })
    .to('.track, .casing', { strokeDashoffset: 0, ...travel }, '<')
    .to('.storm', { motionPath: { path: map.track }, ...travel }, '<')
    .to('.spin', { rotation: -720, svgOrigin: '0 0', duration: travel.duration, ease: 'power2.out' }, '<')
    .addLabel('step3')
    .to('.posts .post', { scale: 1, ...pop, stagger: { amount: 2, from: 'random', ease: flood } })
    .to('.legend .one', { opacity: 1, ...enter }, '<')
    .addLabel('step4')
    .to('.posts', { opacity: 0.35, ...exit })
    .to('.requests, .requests .post', { opacity: 1, ...enter }, '<')
    .to('.legend .one', { opacity: 0, ...exit }, '<')
    .to('.legend .sorted', { opacity: 1, ...enter }, '<')
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="risk-map">
    <div class="figures">
      <div class="figure rank">
        <div class="number">1st</div>
        <div class="statement">Highest disaster risk<br>among 193 countries</div>
      </div>
      <div class="figure cyclones">
        <div class="number">8–9</div>
        <div class="statement">Tropical cyclones cross<br>the country each year</div>
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

      <g class="legend">
        <g class="one">
          <circle class="post" cx="300" cy="12" r="5" />
          <text x="312" y="18">One post</text>
        </g>
        <g class="sorted">
          <circle class="post request" cx="300" cy="12" r="5" />
          <text x="312" y="18">Requests or urgent needs</text>
          <circle class="post faded" cx="300" cy="40" r="5" />
          <text x="312" y="46">Other posts</text>
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

svg {
  width: calc(100% + var(--margin-x));
  height: calc(100% + var(--margin-bottom));
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

footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}

</style>
