<script setup lang="ts">
import { archive, cells, hinge, lock, post } from '../shared/archive'

defineProps<{ locked?: boolean }>()
</script>

<template>
  <g class="archive">
    <rect class="archive-outline" :x="archive.x" :y="archive.y" :width="archive.width" :height="archive.height" rx="14" />
    <text class="archive-name" :x="archive.x + 24" :y="archive.y + 44">X API</text>
    <text class="archive-detail" :x="archive.x + 24" :y="archive.y + 68">Full-archive search</text>

    <path
      class="lock-shackle" :d="`M ${lock.x - 9} ${lock.y + 2} V ${lock.y - 12} a 9 9 0 0 1 18 0 V ${lock.y + 2}`"
      :transform="locked ? undefined : `rotate(${lock.open} ${hinge.x} ${hinge.y})`"
    />
    <rect class="lock-body" :x="lock.x - 14" :y="lock.y" width="28" height="22" rx="4" />
    <circle class="lock-hole" :cx="lock.x" :cy="lock.y + 10" r="3" />

    <g v-for="(cell, index) in cells" :key="index" class="archive-cell" :class="`cell-${index}`" :transform="`translate(${cell.x} ${cell.y})`">
      <rect class="empty-slot" :width="post.width" :height="post.height" rx="6" />
      <g class="card-motion">
        <PostCard />
      </g>
    </g>
  </g>
</template>

<style scoped>
.archive-outline {
  fill: var(--color-surface);
  stroke: var(--color-data);
  stroke-width: 2.4;
}

.archive-name {
  font-size: 24px;
  font-weight: 700;
  fill: var(--color-ink);
}

.archive-detail {
  font-size: 15px;
  fill: var(--color-muted);
}

.lock-shackle {
  fill: none;
  stroke: var(--color-data);
  stroke-width: 3;
}

.lock-body {
  fill: var(--color-data);
}

.lock-hole {
  fill: var(--color-surface);
}

.empty-slot {
  fill: none;
  stroke: var(--color-removed);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
  opacity: 0;
}
</style>
