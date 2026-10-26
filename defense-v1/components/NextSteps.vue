<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Carlo's actions right after approval, then Prinz's two lanes, which run at the same time.
// Drawn like the flow diagram on the next slide, which starts where these lanes end.
// A time arrow along the top runs from approval into the next few weeks

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }

// Positions, in the coordinates of the 884 by 456 drawing
const left = { width: 170 }
const lanesX = 214
const gap = 24
const boxHeight = 60
const axisY = 64

const lanes = [
  { key: 'data', name: 'Data', after: 'X API access', y: 196, steps: ['Pilot search', 'Set limits', 'Collect, clean', 'Filter cyclones'] },
  { key: 'labeling', name: 'Labeling', after: 'Revise chapters', y: 360, steps: ['Draft guidelines', 'Set up doccano', 'Practice batch'] },
]

// Both lanes fill the same width, so each box is as wide as its lane allows
const boxes = (count: number) => {
  const width = (884 - lanesX - gap * (count - 1)) / count
  return Array.from({ length: count }, (_, index) => ({ x: lanesX + index * (width + gap), width }))
}

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.after, .lane', { opacity: 0 })
  gsap.set('.lane > *', { opacity: 0 })

  timeline
    .addLabel('step0')
    // Carlo. Right after approval
    .to('.after', { opacity: 1, ...enter, stagger: 0.15 })
    .addLabel('step1')
    // Prinz. The data lane, one step after another. Earlier steps stay at full strength
    .set('.lane.data', { opacity: 1 })
    .to('.data > *', { opacity: 1, ...enter, stagger: 0.12 })
    .addLabel('step2')
    // At the same time, the labeling lane
    .set('.lane.labeling', { opacity: 1 })
    .to('.labeling > *', { opacity: 1, ...enter, stagger: 0.12 })
    .addLabel('step3')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 456" width="884" height="456">
    <text class="tag" x="0" y="34">After approval</text>
    <text class="tag" :x="lanesX" y="34">The next two to four weeks</text>

    <!-- Time runs left to right. The diamond is approval, like the final defense on the schedule -->
    <path class="approval" :d="`M 8 ${axisY} l 8 -8 l 8 8 l -8 8 z`" />
    <path class="axis" :d="`M 30 ${axisY} H 872`" />
    <path class="axis-head" :d="`M 884 ${axisY} l -12 -6 v 12 z`" />
    <path class="axis" :d="`M ${lanesX - 25} ${axisY - 8} V ${axisY + 8}`" />

    <g v-for="lane in lanes" :key="`after-${lane.key}`" class="after">
      <rect class="box" x="0" :y="lane.y - boxHeight / 2" :width="left.width" :height="boxHeight" rx="12" />
      <text class="box-text" :x="left.width / 2" :y="lane.y + 6" text-anchor="middle">{{ lane.after }}</text>
    </g>

    <g v-for="lane in lanes" :key="lane.key" class="lane" :class="lane.key">
      <text class="tag" :x="lanesX" :y="lane.y - 48">{{ lane.name }}</text>
      <text v-if="lane.key === 'labeling'" class="together" x="884" :y="lane.y - 48" text-anchor="end">At the same time</text>
      <g v-for="(box, index) in boxes(lane.steps.length)" :key="lane.steps[index]">
        <template v-if="index > 0">
          <path class="line" :d="`M ${box.x - gap + 2} ${lane.y} H ${box.x - 12}`" />
          <path class="head" :d="`M ${box.x - 2} ${lane.y} l -11 -6 v 12 z`" />
        </template>
        <rect class="box" :x="box.x" :y="lane.y - boxHeight / 2" :width="box.width" :height="boxHeight" rx="12" />
        <text class="box-text" :x="box.x + box.width / 2" :y="lane.y + 6" text-anchor="middle">{{ lane.steps[index] }}</text>
      </g>
    </g>
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

.together {
  font-size: var(--text-2xs);
  font-style: italic;
  fill: var(--color-muted);
}

.approval {
  fill: var(--color-ink);
}

.axis {
  fill: none;
  stroke: var(--color-muted);
  stroke-width: 2;
}

.axis-head {
  fill: var(--color-muted);
}

.box {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

/* A little under the deck's smallest size, so the longest step fits on one line in its box */
.box-text {
  font-size: 17px;
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
</style>
