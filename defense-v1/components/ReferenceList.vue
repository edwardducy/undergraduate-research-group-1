<script setup lang="ts">
import { computed } from 'vue'
// Built by scripts/references.py from the works cited in the presentation, in APA style
import references from '../data/references.json'

const props = defineProps<{ page: number }>()

// The list is split into a set number of pages of about the same length, keeping the
// alphabetical order. If this number changes, the reference pages in slides.md must change with it
const pageCount = 2
const size = (entry: string) => entry.replace(/<[^>]+>/g, '').length + 120

const pages = computed(() => {
  const total = references.reduce((sum, entry) => sum + size(entry), 0)
  const result: string[][] = Array.from({ length: pageCount }, () => [])
  let before = 0
  for (const entry of references) {
    // Each entry goes on the page where its middle falls
    const page = Math.min(pageCount - 1, Math.floor((before + size(entry) / 2) / (total / pageCount)))
    result[page].push(entry)
    before += size(entry)
  }
  return result
})

const entries = computed(() => pages.value[props.page - 1] ?? [])
</script>

<template>
  <div class="references">
    <h3>{{ pages.length > 1 ? `References, ${page} of ${pages.length}` : 'References' }}</h3>
    <div class="columns">
      <!-- Pandoc's own markup, which only marks titles and volumes in italics -->
      <p v-for="(entry, index) in entries" :key="index" class="entry" v-html="entry" />
    </div>
  </div>
</template>

<style scoped>
/* The slide's content area is 456 pixels tall, inside the layout's padding */
.references {
  height: 456px;
  display: flex;
  flex-direction: column;
}

h3 {
  margin-bottom: var(--space-1);
}

.columns {
  flex: 1;
  min-height: 0;
  column-count: 2;
  column-gap: var(--space-4);
  /* Fill the left column first, so any overflow spills into a third column that can be measured */
  column-fill: auto;
}

/* APA hanging indent */
.entry {
  break-inside: avoid;
  margin: 0 0 6px;
  padding-left: 1.5em;
  text-indent: -1.5em;
  font-size: 0.72rem;
  line-height: 1.3;
  overflow-wrap: anywhere;
}
</style>
