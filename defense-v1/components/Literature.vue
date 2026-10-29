<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { addStep, useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit } from '../shared/motion'

const studies = [
  {
    area: 'Multi-task disaster models',
    name: 'Ali et al. (2026)',
    did: 'One encoder with a part for each task, trained on English disaster posts',
    open: 'Other languages, and PCGrad combined with uncertainty weighting',
    ours: 'Same design on Taglish, with the combined optimizer among our 13',
  },
  {
    area: 'Philippine disaster systems',
    name: 'Caoile and Aviles (2025)',
    did: 'A large language model sorts Taglish comments and picks out details',
    open: 'Both outputs come from a large language model',
    ours: 'A smaller model trained on labeled Taglish posts',
  },
  {
    area: 'Optimizer benchmarks',
    name: 'Kurin et al. (2022)',
    did: 'Equal weighting against advanced optimizers, on scores and training time',
    open: 'Settings where equal weighting does worse, beyond images and robot tasks',
    ours: 'Taglish text, with one task per post and one per word',
  },
]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()
  const ink = { color: colour('ink'), ...enter }

  gsap.set('.cell', { color: colour('faint') })
  gsap.set('.kyle', { opacity: 0 })

  timeline.addLabel('step0')
  studies.forEach((_, index) => {
    timeline.to(`.study-${index} .name-cell, .study-${index} .did`, ink)
    addStep(timeline)
  })

  timeline
    .to('.prinz', { opacity: 0, ...exit })
    .to('.kyle', { opacity: 1, ...enter })
  addStep(timeline)

  studies.forEach((_, index) => {
    for (const column of ['open', 'ours']) {
      timeline.to(`.study-${index} .${column}`, ink)
      addStep(timeline)
    }
  })
})
</script>

<template>
  <div ref="root" class="literature">
    <div class="row heading">
      <span>Study</span>
      <div class="swap">
        <span class="prinz">What it did</span>
        <div class="kyle pair">
          <span>Left open</span>
          <span>Our study</span>
        </div>
      </div>
    </div>

    <div v-for="(study, index) in studies" :key="study.name" class="row" :class="`study-${index}`">
      <div class="cell name-cell">
        <span class="area">{{ study.area }}</span>
        <b>{{ study.name }}</b>
      </div>
      <div class="swap">
        <span class="cell prinz did">{{ study.did }}</span>
        <div class="kyle pair">
          <span class="cell open">{{ study.open }}</span>
          <span class="cell ours">{{ study.ours }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.literature {
  display: grid;
  align-content: center;
  height: 456px;
}

.row {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: var(--space-4);
  padding: 1.25rem 0;
  font-size: var(--text-xs);
  line-height: var(--leading-list);
}

.row + .row {
  border-top: 1px solid var(--color-faint);
}

.heading {
  padding-top: 0;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--color-ink);
  font-size: var(--text-2xs);
  font-weight: 600;
  line-height: var(--leading-tight);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
}

.heading + .row {
  border-top: none;
}

.swap {
  display: grid;
}

.swap > * {
  grid-area: 1 / 1;
}

.pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

.area {
  display: block;
  margin-bottom: 0.375rem;
  font-size: 0.95rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

b {
  font-weight: 600;
}
</style>
