<script setup lang="ts">
import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { ref, watch } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

const rows = [
  { key: 'single', name: 'Most Philippine studies reviewed', note: 'Train a model for only one task' },
  { key: 'llm', name: 'Urdaneta City system', note: 'Sends posts to a large language model', caveat: true },
  { key: 'small', name: 'Smaller trained model', note: 'Needs Taglish posts labeled for both tasks' },
]

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const draw = { duration: 0.5, ease: 'power2.out' }

const root = ref<HTMLElement>()
const { $clicks } = useSlideContext()

// The first row's check moves back and forth, because each study covers one task or the other
let swap: gsap.core.Timeline | undefined

function loopSwap(step: number) {
  if (step >= 1)
    swap?.play()
  else
    swap?.pause(0)
}

useClickTimeline(root, (timeline) => {
  const style = getComputedStyle(document.documentElement)
  const ink = style.getPropertyValue('--color-ink').trim()
  const negative = style.getPropertyValue('--color-negative').trim()

  gsap.set('.head, .row, .datasets .label, .card, .highlight, .first', { opacity: 0 })
  // The name starts black like the others, then on its own click turns red as a line crosses it out
  gsap.set('.llm .crossed', { color: ink })
  gsap.set('.llm .strike', { scaleX: 0, transformOrigin: '0% 50%' })
  gsap.set('.check, .cross', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.single .details .mark', { opacity: 0.15 })

  swap = gsap.timeline({ repeat: -1, paused: true })
    .to('.single .details .mark', { opacity: 1, duration: 0.3 }, 1.2)
    .to('.single .type .mark', { opacity: 0.15, duration: 0.3 }, '<')
    .to('.single .type .mark', { opacity: 1, duration: 0.3 }, '+=1.2')
    .to('.single .details .mark', { opacity: 0.15, duration: 0.3 }, '<')

  timeline
    .addLabel('step0')
    .to('.head, .single', { opacity: 1, ...enter })
    .to('.single .check', { strokeDashoffset: 0, ...draw }, '<0.2')
    .addLabel('step1')
    .to('.llm', { opacity: 1, ...enter })
    .to('.llm .check', { strokeDashoffset: 0, ...draw, stagger: 0.15 }, '<0.2')
    .addLabel('step2')
    .to('.llm .crossed', { color: negative, ...enter })
    .to('.llm .strike', { scaleX: 1, ...draw }, '<')
    .addLabel('step3')
    .to('.small', { opacity: 1, ...enter })
    .to('.small .check', { strokeDashoffset: 0, ...draw, stagger: 0.15 }, '<0.2')
    .addLabel('step4')
    .to('.datasets .label, .english', { opacity: 1, ...enter })
    .to('.english .check', { strokeDashoffset: 0, ...draw }, '<0.2')
    .addLabel('step5')
    .to('.taglish', { opacity: 1, ...enter })
    .to('.taglish .cross', { strokeDashoffset: 0, ...draw }, '<0.2')
    .addLabel('step6')
    .to('.highlight, .first', { opacity: 1, ...enter })
    .addLabel('step7')

  loopSwap($clicks.value)
})

watch($clicks, loopSwap)
</script>

<template>
  <div ref="root" class="dataset-gap">
    <div class="table">
      <div class="head">
        <span />
        <span class="tag classification">Report type</span>
        <span class="tag ner">Key details</span>
      </div>

      <div v-for="row in rows" :key="row.key" class="row" :class="row.key">
        <div>
          <div class="name">
            <span v-if="row.caveat" class="crossed">{{ row.name }}<span class="strike" /></span>
            <template v-else>{{ row.name }}</template>
          </div>
          <div class="note">{{ row.note }}</div>
        </div>
        <div class="cell type classification">
          <svg class="mark" viewBox="0 0 28 28"><path class="check" d="M 5 15 L 11 21 L 23 7" pathLength="1" /></svg>
        </div>
        <div class="cell details ner">
          <svg class="mark" viewBox="0 0 28 28"><path class="check" d="M 5 15 L 11 21 L 23 7" pathLength="1" /></svg>
        </div>
      </div>
    </div>

    <div class="datasets">
      <div class="tag label">Datasets labeled for both tasks</div>
      <div class="card english">
        <svg class="mark" viewBox="0 0 28 28"><path class="check" d="M 5 15 L 11 21 L 23 7" pathLength="1" /></svg>
        <span class="name">English</span>
      </div>
      <div class="card taglish">
        <div class="highlight" />
        <svg class="mark" viewBox="0 0 28 28"><path class="cross" d="M 7 7 L 21 21 M 21 7 L 7 21" pathLength="1" /></svg>
        <div>
          <div class="name">Taglish</div>
          <div class="note">None among the datasets reviewed</div>
        </div>
      </div>
      <div class="first">The first step is to build this dataset</div>
    </div>
  </div>
</template>

<style scoped>
.dataset-gap {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.head,
.row {
  display: grid;
  grid-template-columns: 1fr 180px 180px;
  align-items: center;
}

.head {
  padding-bottom: var(--space-1);
  border-bottom: 2px solid var(--color-land);
}

.head .tag {
  text-align: center;
}

.row {
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--color-land);
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.label {
  color: var(--color-muted);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
  line-height: var(--leading-tight);
}

.note {
  font-size: var(--text-2xs);
  color: var(--color-muted);
  margin-top: 0.15rem;
}

.cell {
  display: flex;
  justify-content: center;
}

/* The Urdaneta City system covers both tasks only by way of a large language model,
   so on its own click its name turns red and a line draws through it */
.crossed {
  position: relative;
  display: inline-block;
}

.strike {
  position: absolute;
  left: -2px;
  right: -2px;
  top: 55%;
  height: 3px;
  background: currentColor;
}

.mark {
  flex: none;
  width: 34px;
  height: 34px;
}

.check,
.cross {
  fill: none;
  stroke: currentColor;
  stroke-width: 3.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.classification {
  color: var(--color-classification);
}

.ner {
  color: var(--color-ner);
}

.datasets {
  display: grid;
  grid-template-columns: auto auto 1fr;
  align-items: center;
  column-gap: var(--space-3);
  row-gap: var(--space-1);
}

.datasets .label {
  grid-column: 1 / 4;
}

.card {
  position: relative;
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-2);
  border: 2px solid var(--color-land);
  border-radius: 12px;
}

.english {
  color: var(--color-ink);
}

.taglish .cross {
  color: var(--color-negative);
}

.highlight {
  position: absolute;
  inset: -2px;
  border: 3px solid var(--color-classification);
  border-radius: 12px;
}

.card .note {
  white-space: nowrap;
}

.first {
  grid-column: 3;
  grid-row: 2;
  font-size: var(--text-base);
  font-weight: 600;
  color: var(--color-classification);
}
</style>
