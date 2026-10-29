<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { approachExamplesX as examplesX, approachModelX as modelX } from '../shared/model'
import { enter, exit, draw, moveShort } from '../shared/motion'

const width = 884

const panels = [
  {
    caption: ['Earlier Philippine', 'studies'],
    pipelines: [
      { x: -66, width: 116, name: 'Model', outputs: ['classification'] },
      { x: 66, width: 116, name: 'Model', outputs: ['ner'] },
    ],
  },
  {
    caption: ['Urdaneta City', 'system'],
    pipelines: [{ x: 0, width: 116, name: 'LLM', outputs: ['classification', 'ner'] }],
  },
  {
    caption: ['This study'],
    pipelines: [{ x: 0, width: 256, name: 'Pretrained encoder', outputs: ['classification', 'ner'] }],
  },
]

const spread = 34

function outputX(outputs: string[], index: number) {
  return outputs.length === 1 ? 0 : (index === 0 ? -spread : spread)
}

function place(index: number, count: number) {
  return (width * (index + 0.5)) / count
}

function bracket(start: number, end: number, top: number) {
  return `M ${start} ${top} V ${top + 6} H ${end} V ${top}`
}

const exampleShift = 63

const examplesY = 186

const dash = 12

function period(path: SVGPathElement) {
  const length = path.getTotalLength()
  return (length - dash) / Math.round((length - dash) / (dash + 6))
}

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.panel, .legend, .example, .english-source', { opacity: 0 })
  gsap.set('.taglish .post, .taglish .bracket', { filter: 'grayscale(0)' })
  panels.forEach((_, index) => gsap.set(`.panel-${index}`, { x: place(index, index + 1) }))

  timeline
    .addLabel('step0')
    .to('.panel-0, .legend', { opacity: 1, ...enter })
    .addLabel('step1')
    .to('.panel-0', { x: place(0, 2), ...moveShort })
    .to('.panel-1', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step2')
    .to('.panel-0', { x: place(0, 3), ...moveShort })
    .to('.panel-1', { x: place(1, 3), ...moveShort }, '<')
    .to('.panel-2', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step3')
    .to('.panel-0, .panel-1, .earlier-sources', { opacity: 0, ...exit })
    .to('.panel-2', { x: width / 2, ...moveShort }, '<0.1')
    .addLabel('step4')
    .to('.panel-2', { x: modelX, ...moveShort })
    .to('.english, .english-source', { opacity: 1, ...enter }, '>-0.1')
    .fromTo('.english .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.2')
    .addLabel('step5')
    .to('.taglish', { opacity: 1, ...enter })
    .to('.taglish', { y: `+=${exampleShift}`, ...moveShort }, '<')
    .to('.english', { y: `-=${exampleShift}`, ...moveShort }, '<')
    .fromTo('.taglish .bracket', {
      strokeDasharray: (_: number, path: SVGPathElement) => `${period(path)} 0`,
      strokeDashoffset: (_: number, path: SVGPathElement) => (period(path) - dash) / 2,
    }, {
      strokeDasharray: (_: number, path: SVGPathElement) => `${dash} ${period(path) - dash}`,
      strokeDashoffset: 0,
      ...draw,
    }, '<0.2')
    .addLabel('step6')
    .to('.taglish .post, .taglish .bracket', { opacity: 0.5, filter: 'grayscale(1)', ...enter })
    .addLabel('step7')
    .set('.panel-2 .model, .panel-2 .name, .panel-2 .classification, .panel-2 .ner', { opacity: 0 }, '+=0.01')
    .to('.example, .legend, .english-source, .panel-2 .post, .panel-2 .input, .panel-2 .tag', { opacity: 0, ...exit })
    .addLabel('step8')
})
</script>

<template>
  <div ref="root" class="approach-diagram">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <g v-for="(panel, index) in panels" :key="index" class="panel" :class="`panel-${index}`">
        <g v-for="pipeline in panel.pipelines" :key="pipeline.x" :transform="`translate(${pipeline.x} 0)`">
          <text class="post" y="32" text-anchor="middle">Post</text>
          <path class="input" d="M 0 46 V 128" />
          <path class="input head" d="M 0 140 l -7 -12 h 14 z" />

          <rect class="model" :x="-pipeline.width / 2" y="144" :width="pipeline.width" height="64" rx="12" />
          <text class="name" y="184" text-anchor="middle">{{ pipeline.name }}</text>

          <g v-for="(output, slot) in pipeline.outputs" :key="output" :class="output">
            <path :d="`M 0 212 V 230 H ${outputX(pipeline.outputs, slot)} V 294`" />
            <path class="head" :d="`M ${outputX(pipeline.outputs, slot)} 306 l -7 -12 h 14 z`" />
            <rect class="swatch" :x="outputX(pipeline.outputs, slot) - 14" y="312" width="28" height="28" rx="4" />
          </g>
        </g>

        <text class="tag" text-anchor="middle">
          <tspan v-for="(line, row) in panel.caption" :key="row" x="0" :y="378 + row * 22">{{ line }}</tspan>
        </text>
      </g>

      <g :transform="`translate(${examplesX} 0)`">
        <g class="example english" :transform="`translate(0 ${examplesY})`">
          <text class="tag" y="-40">English post</text>
          <text class="post">Flooding in Houston, we need water!</text>
          <path class="bracket ner" :d="bracket(125, 216, 9)" />
          <path class="bracket ner" :d="bracket(266, 387, 9)" />
          <path class="bracket classification" :d="bracket(0, 393, 27)" />
        </g>
        <g class="example taglish" :transform="`translate(0 ${examplesY})`">
          <text class="tag" y="-40">Taglish post</text>
          <text class="post">Baha na sa Marikina, need ng tubig!</text>
          <path class="bracket ner" :d="bracket(125, 217, 9)" />
          <path class="bracket ner" :d="bracket(229, 378, 9)" />
          <path class="bracket classification" :d="bracket(0, 384, 27)" />
        </g>
      </g>

      <g class="legend">
        <rect class="swatch classification" x="93" y="430" width="14" height="14" rx="3" />
        <text class="tag" x="115" y="443">Humanitarian classification</text>
        <rect class="swatch ner" x="477" y="430" width="14" height="14" rx="3" />
        <text class="tag" x="499" y="443">Named entity recognition</text>
      </g>
    </svg>

    <footer class="earlier-sources">Sources: Ermino et al. (2022), Somido &amp; Reyes (2023), Yute et al. (2021), Caoile &amp; Aviles (2025)</footer>
    <footer class="english-source">Source: Ali et al. (2026)</footer>
  </div>
</template>

<style scoped>
.approach-diagram {
  position: relative;
}

.post {
  font-size: var(--text-base);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

path {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head {
  fill: var(--color-ink);
  stroke: none;
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.swatch {
  fill: currentColor;
}

.classification {
  color: var(--color-classification);
}

.ner {
  color: var(--color-ner);
}

.bracket {
  stroke: currentColor;
  stroke-width: 2.5;
}

footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
