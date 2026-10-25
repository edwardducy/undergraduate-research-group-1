<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }

// Reasons not being discussed fade back but stay readable, as on slide 4
const dim = 0.35

// The run counts from Chapter 3
const runs = { tuning: 92, main: 51 }

// Illustration only. Peak memory of each model in the pilot, all under the laptop's 8 GB
const memory = [46, 58, 42, 60, 56, 49, 61, 39]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.reason', { opacity: dim })
  gsap.set('.reason .detail', { opacity: 0 })

  timeline.addLabel('step0')
  ;['pilot', 'workload', 'budget'].forEach((key, index) => {
    timeline
      .to(`.reason:not(.${key})`, { opacity: dim, ...exit })
      .to(`.reason.${key}`, { opacity: 1, ...enter }, '<')
      .to(`.reason.${key} .detail`, { opacity: 1, ...enter }, '<0.1')
      .addLabel(`step${index + 1}`)
  })
})
</script>

<template>
  <div ref="root" class="reasons">
    <section class="reason pilot">
      <p class="title">The training pilot already ran</p>
      <div class="detail">
        <svg viewBox="0 0 196 100" width="196" height="100">
          <rect v-for="(height, index) in memory" :key="index" class="memory-bar" :x="6 + index * 24" :y="96 - height" width="16" :height="height" rx="2" />
          <path class="limit" d="M 0 30 H 196" />
          <text class="limit-label" x="196" y="22" text-anchor="end">8 GB</text>
          <path class="baseline" d="M 0 96 H 196" />
        </svg>
        <p class="text">Every model fits in the laptop's memory, after one change shown with the risks</p>
      </div>
    </section>

    <section class="reason workload">
      <p class="title">The workload is fixed in advance</p>
      <div class="detail">
        <div class="runs">
          <div class="run-group">
            <div class="squares"><i v-for="run in runs.tuning" :key="run" /></div>
            <span class="count">{{ runs.tuning }} tuning runs</span>
          </div>
          <div class="run-group">
            <div class="squares"><i v-for="run in runs.main" :key="run" class="main" /></div>
            <span class="count">{{ runs.main }} main runs</span>
          </div>
        </div>
        <p class="text">So the training time can be planned</p>
      </div>
    </section>

    <section class="reason budget">
      <p class="title">X API spending is capped</p>
      <div class="detail">
        <p class="amount">US$200</p>
        <div class="split"><i v-for="part in 5" :key="part" /></div>
        <p class="text">A pilot search helps spread it across the cyclones</p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.reasons {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-2);
  height: 384px;
}

.reason {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-2);
  border-radius: 12px;
  background: var(--color-land);
}

.title {
  margin: 0;
  font-size: var(--text-base);
  font-weight: 700;
  line-height: var(--leading-tight);
}

.detail {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  flex: 1;
}

.text {
  margin: auto 0 0;
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
}

.memory-bar {
  fill: var(--color-subtle);
}

.limit {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 5 4;
}

.limit-label {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}

.baseline {
  stroke: var(--color-subtle);
  stroke-width: 1.5;
}

.runs {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.squares {
  display: grid;
  grid-template-columns: repeat(23, 1fr);
  gap: 2px;
}

.squares i {
  aspect-ratio: 1;
  border-radius: 1px;
  background: var(--color-subtle);
}

.squares i.main {
  background: var(--color-ink);
}

.count {
  font-size: var(--text-2xs);
  font-weight: 600;
}

.amount {
  margin: 0;
  font-size: var(--text-4xl);
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}

/* The budget split into parts, one for each cyclone */
.split {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 3px;
}

.split i {
  height: 14px;
  border-radius: 3px;
  background: var(--color-ink);
}
</style>
