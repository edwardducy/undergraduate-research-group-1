<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }

const dim = 0.35

// Carlo's actions right after approval, then Prinz's two lanes, which run at the same time
const afterApproval = ['Revise Chapters 1 to 3', 'Secure full-archive access to the X API']
const lanes = [
  { key: 'data', name: 'Data', steps: ['Pilot search', 'Set a limit for each cyclone', 'Collect and clean the posts', 'Keep cyclones with 400+ Taglish posts'] },
  { key: 'labeling', name: 'Labeling', steps: ['Draft guidelines with Taglish examples', 'Set up doccano', 'Label a practice batch'] },
]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.action, .step, .arrow, .together', { opacity: 0 })

  timeline
    .addLabel('step0')
    // Carlo. Right after approval
    .to('.action', { opacity: 1, ...enter, stagger: 0.2 })
    .addLabel('step1')
    // Prinz. The data lane, one step after another
    .to('.action', { opacity: dim, ...exit })
    .to('.data .step, .data .arrow', { opacity: 1, ...enter, stagger: 0.15 }, '<0.1')
    .addLabel('step2')
    // At the same time, the labeling lane
    .to('.together', { opacity: 1, ...enter })
    .to('.labeling .step, .labeling .arrow', { opacity: 1, ...enter, stagger: 0.15 }, '<0.1')
    .addLabel('step3')
})
</script>

<template>
  <div ref="root" class="next-steps">
    <section class="approval">
      <div class="tag">Right after approval</div>
      <div v-for="action in afterApproval" :key="action" class="action">{{ action }}</div>
    </section>

    <section class="weeks">
      <div class="tag">The next two to four weeks</div>
      <div v-for="lane in lanes" :key="lane.key" class="lane" :class="lane.key">
        <div class="lane-name">{{ lane.name }}</div>
        <template v-for="(step, index) in lane.steps" :key="step">
          <span v-if="index > 0" class="arrow">→</span>
          <div class="step">{{ step }}</div>
        </template>
        <div v-if="lane.key === 'labeling'" class="together">At the same time</div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.next-steps {
  display: grid;
  grid-template-columns: 200px 1fr;
  column-gap: var(--space-4);
  height: 384px;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-2);
}

.approval {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding-right: var(--space-3);
  border-right: 2px solid var(--color-land);
}

.approval .tag {
  margin-bottom: 0;
}

.action {
  padding: 12px 14px;
  border-radius: 10px;
  border: 2px solid var(--color-ink);
  font-size: var(--text-xs);
  font-weight: 600;
  line-height: var(--leading-list);
}

.weeks {
  display: flex;
  flex-direction: column;
}

/* A lane name, then four steps with arrows between them */
.lane {
  position: relative;
  display: grid;
  grid-template-columns: 118px 1fr 22px 1fr 22px 1fr 22px 1fr;
  align-items: center;
  min-height: 120px;
}

.lane + .lane {
  margin-top: var(--space-4);
}

.lane-name {
  font-size: var(--text-base);
  font-weight: 700;
}

.step {
  display: grid;
  place-items: center;
  min-height: 104px;
  padding: 10px;
  border-radius: 10px;
  background: var(--color-land);
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
  text-align: center;
}

.arrow {
  text-align: center;
  color: var(--color-muted);
  font-size: var(--text-base);
}

.together {
  position: absolute;
  left: 118px;
  top: calc(-1 * var(--space-4));
  font-size: var(--text-2xs);
  font-style: italic;
  color: var(--color-muted);
}
</style>
