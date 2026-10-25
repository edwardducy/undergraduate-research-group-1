<script setup lang="ts">
import { onSlideEnter } from '@slidev/client'
import gsap from 'gsap'
import { onMounted, onUnmounted, ref } from 'vue'

// A block sweeps in over the title, the title appears under it, and the block sweeps out the other side.
// It plays each time the slide opens, without taking a click
const root = ref<HTMLElement>()

const sweep = { duration: 0.4, ease: 'power2.inOut' }

let context: gsap.Context
let reveal: gsap.core.Timeline

onMounted(() => {
  context = gsap.context(() => {
    reveal = gsap.timeline({ paused: true })
      .set('.text', { opacity: 0 })
      .set('.block', { scaleX: 0, transformOrigin: '0% 50%' })
      // Wait for the slide transition to settle first
      .to('.block', { scaleX: 1, ...sweep }, 0.35)
      .set('.text', { opacity: 1 })
      .set('.block', { transformOrigin: '100% 50%' })
      .to('.block', { scaleX: 0, ...sweep })

    // Shown in full until the slide opens, so the overview and exports still show the title
    reveal.progress(1)
  }, root.value)
})

// Restarting hides the title at once, so it stays hidden while the slide transitions in
onSlideEnter(() => reveal?.restart())

onUnmounted(() => context?.revert())
</script>

<template>
  <h1 ref="root" class="reveal-title">
    <span class="frame">
      <span class="text"><slot /></span>
      <span class="block" />
    </span>
  </h1>
</template>

<style scoped>
.frame {
  position: relative;
  display: inline-block;
}

.block {
  position: absolute;
  inset: -0.08em -0.12em;
  /* The block takes the colour of the text it reveals */
  background: currentColor;
}
</style>
