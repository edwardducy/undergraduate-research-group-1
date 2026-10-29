<script setup lang="ts">
import { useSlideContext } from '@slidev/client'

const { $frontmatter } = useSlideContext()
</script>

<template>
  <div class="slidev-layout default">
    <p v-if="$frontmatter.section" class="section">{{ $frontmatter.section }}</p>
    <slot />
  </div>
</template>

<style scoped>
.default {
  position: relative;
}

.section {
  position: absolute;
  top: var(--margin-section);
  left: var(--margin-x);
  margin: 0;
  font-size: var(--text-2xs);
  font-weight: 400;
  line-height: var(--leading-tight);
  color: var(--color-subtle);
}
</style>
