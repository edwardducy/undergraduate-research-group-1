<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter } from '../shared/motion'

const covered = [
  'Taglish typhoon posts, 2020–2026, mainly from X',
  'About 3,900 posts labeled for both tasks',
  'One encoder, dataset, and training process for all optimizers',
  'A prototype web application',
]
const excluded = [
  'Comparison with large language models',
  'Deployment in real disasters',
  'Testing with disaster office staff',
]
const lists = [
  { heading: 'Covered', items: covered, first: 0 },
  { heading: 'Not covered', items: excluded, first: covered.length },
]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.item', { color: colour('faint') })

  timeline.addLabel('step0')
  for (let index = 0; index < covered.length + excluded.length; index++) {
    timeline
      .to(`.item-${index}`, { color: colour('ink'), ...enter })
      .addLabel(`step${index + 1}`)
  }
})
</script>

<template>
  <div ref="root" class="scope">
    <section v-for="list in lists" :key="list.heading">
      <h3>{{ list.heading }}</h3>
      <ul>
        <li v-for="(item, index) in list.items" :key="item" class="item" :class="`item-${list.first + index}`">{{ item }}</li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.scope {
  display: grid;
  grid-template-columns: 1fr 1fr;
  align-content: center;
  gap: 56px;
  height: 456px;
}

h3 {
  margin-bottom: var(--space-2);
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--color-ink);
}

ul {
  display: grid;
  gap: var(--space-2);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  line-height: var(--leading-list);
}
</style>
