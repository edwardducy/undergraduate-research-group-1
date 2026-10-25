<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const pop = { duration: 0.25, ease: 'back.out(3)' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }

// Checkpoints not yet reached stay faded
const dim = 0.35

// The same green as the chosen optimizer on slide 8, marking a checkpoint as passed
const passed = '#15803d'

const checkpoints = [
  { key: 'agreement', title: 'The practice batch meets the agreement targets' },
  { key: 'size', title: 'The selected cyclones give about 3,900 posts' },
  { key: 'facebook', title: 'A decision on adding Facebook posts' },
]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.checkpoint', { opacity: dim })
  gsap.set('.tick', { opacity: 0, scale: 0.5, transformOrigin: '50% 50%' })
  gsap.set('.opened', { opacity: 0 })

  timeline.addLabel('step0')

  // Each checkpoint is met in turn and stays checked
  checkpoints.forEach(({ key }, index) => {
    timeline
      .to(`.checkpoint.${key}`, { opacity: 1, ...enter })
      .to(`.checkpoint.${key} .tick`, { opacity: 1, scale: 1, ...pop }, '<0.2')
      .to(`.checkpoint.${key} .badge`, { backgroundColor: passed, borderColor: passed, ...enter }, '<')
      .addLabel(`step${index + 1}`)
  })

  // With all three met, the gate opens and full labeling starts
  timeline
    .to('.door.left', { xPercent: -100, ...moveLong })
    .to('.door.right', { xPercent: 100, ...moveLong }, '<')
    .to('.opened', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step4')
})
</script>

<template>
  <div ref="root" class="gate-page">
    <div class="checkpoints">
      <section v-for="checkpoint in checkpoints" :key="checkpoint.key" class="checkpoint" :class="checkpoint.key">
        <span class="badge">
          <svg class="tick" viewBox="-12 -12 24 24" width="22" height="22"><path d="M -6 0 L -2 5 L 7 -5" /></svg>
        </span>
        <p class="title">{{ checkpoint.title }}</p>

        <!-- The agreement targets from slide 7 -->
        <div v-if="checkpoint.key === 'agreement'" class="detail">
          <div class="target classification"><strong>κ ≥ .61</strong><span>categories</span></div>
          <div class="target ner"><strong>F1 ≥ .74</strong><span>entities</span></div>
        </div>

        <!-- The pool of posts from slide 7, each square about 78 posts -->
        <div v-else-if="checkpoint.key === 'size'" class="detail grid">
          <i v-for="square in 50" :key="square" />
        </div>

        <!-- The fallback from slide 10 -->
        <div v-else class="detail">
          <p class="note">Based on how many Taglish posts X gives</p>
        </div>
      </section>
    </div>

    <section class="gate">
      <div class="gate-label">Full labeling</div>
      <div class="frame">
        <p class="opened">Full labeling starts</p>
        <div class="door left" />
        <div class="door right" />
      </div>
    </section>
  </div>
</template>

<style scoped>
.gate-page {
  display: grid;
  grid-template-columns: 1fr 210px;
  column-gap: var(--space-4);
  height: 384px;
}

.checkpoints {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-2);
}

.checkpoint {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-2);
  border-radius: 12px;
  background: var(--color-land);
}

.badge {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 2px solid var(--color-subtle);
  background: var(--color-surface);
}

.tick {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.title {
  margin: 0;
  font-size: var(--text-xs);
  font-weight: 600;
  line-height: var(--leading-list);
}

.detail {
  margin-top: auto;
}

.target {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 6px 10px;
  margin-top: 6px;
  border-radius: 6px;
  font-size: var(--text-2xs);
  font-weight: 600;
  background: var(--color-surface);
}

.target strong {
  white-space: nowrap;
}

.target span {
  font-weight: 400;
}

.target.classification {
  color: var(--color-classification);
}

.target.ner {
  color: var(--color-ner);
}

.grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
}

.grid i {
  aspect-ratio: 1;
  border-radius: 2px;
  background: var(--color-subtle);
}

.note {
  margin: 0;
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
  color: var(--color-muted);
}

.gate {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.gate-label {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
}

/* Two doors cover the frame, then slide apart */
.frame {
  position: relative;
  flex: 1;
  overflow: hidden;
  border: 2px solid var(--color-ink);
  border-radius: 12px;
  display: grid;
  place-items: center;
}

.door {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 50%;
  background: var(--color-ink);
}

.door.left {
  left: 0;
  border-right: 1px solid var(--color-surface);
}

.door.right {
  right: 0;
  border-left: 1px solid var(--color-surface);
}

.opened {
  margin: 0;
  padding: var(--space-2);
  text-align: center;
  font-size: var(--text-md);
  font-weight: 700;
  color: #15803d;
}
</style>
