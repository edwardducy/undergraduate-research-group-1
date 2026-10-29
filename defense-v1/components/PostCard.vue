<template>
  <g>
    <rect class="card" width="56" height="40" rx="6" />
    <circle class="mark" cx="11" cy="12" r="5" />
    <path class="text strong" d="M 21 12 H 47" />
    <path class="text" d="M 8 25 H 48 M 8 32 H 38" />
  </g>
</template>

<style scoped>
.card {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
  stroke-width: 1.5;
}

.mark {
  fill: var(--color-data);
}

.text {
  stroke: var(--color-data);
  stroke-width: 2;
  opacity: 0.4;
}

.text.strong {
  opacity: 0.6;
}
</style>
