<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Picks up where the two lanes of the next steps slide end. Each lane ends at a milestone, drawn as a
// diamond like the final defense on the schedule. A "No" leads to a backup that loops back to check
// again, and two "Yes" answers lead to full labeling. Drawn like the flow diagrams on slides 2 and 4

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }

// The lane not being discussed fades back but stays readable
const dim = 0.3
// Marks the milestones as met, the same green as on the schedule and optimizer slides
const met = '#15803d'

const lanes = [
  { key: 'data', tag: 'Data', start: 'Cleaned posts', question: ['Enough', 'posts?'], backup: 'Add Facebook posts', y: 66 },
  { key: 'labeling', tag: 'Labeling', start: 'Practice batch', question: ['Annotators', 'agree?'], backup: 'Revise the guidelines', y: 256 },
]

// Shared positions, in the coordinates of the 884 by 384 drawing
const start = { width: 220, height: 60 }
const diamond = { x: 392, halfWidth: 100, halfHeight: 50 }
const backup = { x: 277, width: 230, height: 46, gap: 80 }
const finish = { x: 692, y: 26, width: 192, height: 270 }

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.question, .branch', { opacity: 0 })

  timeline.addLabel('step0')

  // The data milestone, and its backup if there are too few posts
  timeline
    .to('.lane.labeling', { opacity: dim, ...exit })
    .to('.data .diamond', { stroke: colour('ink'), ...enter }, '<')
    .to('.data .question', { opacity: 1, ...enter }, '<0.1')
    .to('.data .branch', { opacity: 1, ...enter }, '>')
    .addLabel('step1')

  // The labeling milestone, and its backup if the annotators do not agree
  timeline
    .to('.lane.data', { opacity: dim, ...exit })
    .to('.lane.labeling', { opacity: 1, ...enter }, '<')
    .to('.labeling .diamond', { stroke: colour('ink'), ...enter }, '<')
    .to('.labeling .question', { opacity: 1, ...enter }, '<0.1')
    .to('.labeling .branch', { opacity: 1, ...enter }, '>')
    .addLabel('step2')

  // Both met. The two "Yes" lines and full labeling turn green
  timeline
    .to('.lane.data', { opacity: 1, ...enter })
    .to('.yes-line', { stroke: met, ...enter }, '<')
    .to('.yes-head, .yes-text', { fill: met, ...enter }, '<')
    .to('.finish-box', { fill: met, stroke: met, ...enter }, '>-0.1')
    .to('.finish-text', { fill: colour('surface'), ...enter }, '<')
    .addLabel('step3')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 384" width="884" height="384">
    <g v-for="lane in lanes" :key="lane.key" class="lane" :class="lane.key">
      <text class="tag" x="0" :y="lane.y - 44">{{ lane.tag }}</text>

      <!-- Where the lane on the next steps slide ends -->
      <rect class="box" x="0" :y="lane.y - start.height / 2" :width="start.width" :height="start.height" rx="12" />
      <text class="box-text" :x="start.width / 2" :y="lane.y + 7" text-anchor="middle">{{ lane.start }}</text>
      <path class="line" :d="`M ${start.width} ${lane.y} H 282`" />
      <path class="head" :d="`M ${diamond.x - diamond.halfWidth} ${lane.y} l -11 -6 v 12 z`" />

      <!-- The milestone -->
      <path
        class="diamond"
        :d="`M ${diamond.x} ${lane.y - diamond.halfHeight} L ${diamond.x + diamond.halfWidth} ${lane.y} L ${diamond.x} ${lane.y + diamond.halfHeight} L ${diamond.x - diamond.halfWidth} ${lane.y} Z`"
      />
      <g class="question">
        <text class="box-text" :x="diamond.x" :y="lane.y - 4" text-anchor="middle">{{ lane.question[0] }}</text>
        <text class="box-text" :x="diamond.x" :y="lane.y + 20" text-anchor="middle">{{ lane.question[1] }}</text>
      </g>

      <!-- If not, the backup, then back to check again -->
      <g class="branch">
        <path class="line" :d="`M ${diamond.x} ${lane.y + diamond.halfHeight} V ${lane.y + backup.gap - 10}`" />
        <path class="head" :d="`M ${diamond.x} ${lane.y + backup.gap} l -6 -11 h 12 z`" />
        <text class="small" :x="diamond.x + 10" :y="lane.y + 70">No</text>
        <rect class="box" :x="backup.x" :y="lane.y + backup.gap" :width="backup.width" :height="backup.height" rx="12" />
        <text class="backup-text" :x="backup.x + backup.width / 2" :y="lane.y + backup.gap + 30" text-anchor="middle">{{ lane.backup }}</text>
        <path class="line" :d="`M ${backup.x} ${lane.y + backup.gap + backup.height / 2} H ${start.width / 2} V ${lane.y + start.height / 2 + 10}`" />
        <path class="head" :d="`M ${start.width / 2} ${lane.y + start.height / 2} l -6 11 h 12 z`" />
        <text class="small" :x="start.width / 2 + 10" :y="lane.y + 74">Check again</text>
      </g>

      <!-- If yes, on to full labeling. Faint until both milestones are met -->
      <path class="yes-line" :d="`M ${diamond.x + diamond.halfWidth} ${lane.y} H ${finish.x - 10}`" />
      <path class="yes-head" :d="`M ${finish.x} ${lane.y} l -11 -6 v 12 z`" />
      <text class="small yes-text" :x="(diamond.x + diamond.halfWidth + finish.x) / 2" :y="lane.y - 12" text-anchor="middle">Yes</text>
    </g>

    <rect class="finish-box" :x="finish.x" :y="finish.y" :width="finish.width" :height="finish.height" rx="12" />
    <text class="finish-text title" :x="finish.x + finish.width / 2" :y="finish.y + finish.height / 2 - 4" text-anchor="middle">Full labeling</text>
    <text class="finish-text sub" :x="finish.x + finish.width / 2" :y="finish.y + finish.height / 2 + 26" text-anchor="middle">starts in December</text>
  </svg>
</template>

<style scoped>
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

.box,
.diamond {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

/* Empty until its milestone is discussed */
.diamond {
  stroke: var(--color-faint);
}

.box-text {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.backup-text {
  font-size: var(--text-2xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.line {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head {
  fill: var(--color-ink);
}

.small {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.yes-line {
  fill: none;
  stroke: var(--color-faint);
  stroke-width: 2.5;
}

.yes-head,
.small.yes-text {
  fill: var(--color-faint);
}

.finish-box {
  fill: var(--color-surface);
  stroke: var(--color-faint);
  stroke-width: 2;
}

.finish-text {
  fill: var(--color-faint);
}

.finish-text.title {
  font-size: var(--text-md);
  font-weight: 700;
}

.finish-text.sub {
  font-size: var(--text-2xs);
}
</style>
