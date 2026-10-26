<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.fill, .actual, .pass, .rule', { opacity: 0 })
  gsap.set('.pass', { scale: 0.6 })

  timeline
    .addLabel('step0')
    // Each test case states an input and the expected result
    .to('.fill', { opacity: 1, ...enter, stagger: 0.15 })
    .addLabel('step1')
    // It passes only if the actual result matches in every case
    .to('.actual', { opacity: 1, ...enter })
    .to('.pass', { opacity: 1, scale: 1, ...pop }, '<0.2')
    .to('.rule', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step2')
})
</script>

<template>
  <div ref="root" class="prototype">
    <div class="card">
      <div class="use-case">
        <span class="tag">Use case</span>
        <span class="title">Correct a label</span>
        <span class="pass">
          <svg viewBox="-12 -12 24 24" width="22" height="22"><path d="M -6 0 L -2 5 L 7 -5" /></svg>
          Pass
        </span>
      </div>
      <div class="columns">
        <div class="column">
          <span class="tag">Input</span>
          <span class="fill">A staff member corrects a label</span>
        </div>
        <div class="column">
          <span class="tag">Expected result</span>
          <span class="fill">The new label replaces the old one and is saved</span>
        </div>
        <div class="column">
          <span class="tag">Actual result</span>
          <span class="actual">The new label replaced the old one and was saved</span>
        </div>
      </div>
    </div>

    <p class="rule">A use case passes only if the actual result matches in every test case</p>

  </div>
</template>

<style scoped>
/* The full height of a slide with no heading, with the card and rule centred in it */
.prototype {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 456px;
}

.card {
  border: 2px solid var(--color-ink);
  border-radius: 12px;
  padding: var(--space-3);
}

.use-case {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-land);
  margin-bottom: var(--space-2);
}

.title {
  font-size: var(--text-lg);
  font-weight: 700;
}

.pass {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  font-size: var(--text-xs);
  font-weight: 700;
  color: var(--color-surface);
  /* The same green as the passed checks on the other slides */
  background: #15803d;
  border-radius: 16px;
  padding: 4px 14px 4px 8px;
}

.pass svg {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.columns {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  column-gap: var(--space-3);
}

.column {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  font-size: var(--text-xs);
  line-height: var(--leading-list);
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
}

.rule {
  margin-top: var(--space-4);
  font-size: var(--text-md);
  font-weight: 600;
}

</style>
