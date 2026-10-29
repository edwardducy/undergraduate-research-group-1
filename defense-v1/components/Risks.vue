<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, moveShort, pop } from '../shared/motion'

const risks = [
  { name: 'Code bugs', plan: 'Test each part, then the whole pipeline' },
  { name: 'Hardware limits', plan: 'Freeze the embedding layer' },
  { name: 'Prototype failures', plan: 'Fix the use case and test again' },
  { name: 'Data and delays', plan: 'Facebook posts, then a stated limitation' },
]
const faded = 0.3

// Risk 1. Placeholder part names until they match the system diagram.
const parts = ['Data prep', 'Model', 'Optimizers', 'Evaluation'].map((name, index) => ({ name, x: index * 112 }))
const sample = { y: 330, from: 66, to: 386 }

// Risk 2. Placeholder memory in GB until the training pilot gives real numbers.
const memory = [6.1, 7.2, 6.8, 9.6, 7.5, 6.4, 10.3, 7.0, 6.6, 11.1, 7.7, 9.2, 6.9]
const chart = { left: 30, right: 285, base: 360, perGb: 22, frozen: 0.68 }
const limit = chart.base - 8 * chart.perGb
const slot = (chart.right - chart.left) / memory.length
const bars = memory.map((gb, index) => ({
  x: chart.left + index * slot + slot * 0.19,
  width: slot * 0.62,
  height: gb * chart.perGb,
}))
const stack = { x: 312, width: 124 }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  const looks = {
    neutral: { name: 'ink', ring: 'ink', fill: 'surface', number: 'ink', shown: 1, tick: 0, plan: 0, note: 'muted' },
    off: { name: 'faint', ring: 'faint', fill: 'surface', number: 'faint', shown: 1, tick: 0, plan: 0, note: 'muted' },
    on: { name: 'ink', ring: 'ink', fill: 'ink', number: 'surface', shown: 1, tick: 0, plan: 1, note: 'muted' },
    done: { name: 'subtle', ring: 'subtle', fill: 'surface', number: 'subtle', shown: 0, tick: 1, plan: 1, note: 'subtle' },
  }
  type Look = keyof typeof looks

  const style = (index: number, name: Look, position: string) => {
    const look = looks[name]
    const item = `.item-${index}`
    timeline
      .to(`${item} .name`, { color: colour(look.name), ...enter }, position)
      .to(`${item} .badge`, { borderColor: colour(look.ring), backgroundColor: colour(look.fill), ...enter }, '<')
      .to(`${item} .number`, { color: colour(look.number), opacity: look.shown, ...enter }, '<')
      .to(`${item} .tick`, { opacity: look.tick, ...enter }, '<')
      .to(`${item} .plan`, { opacity: look.plan, color: colour(look.note), ...enter }, '<')
  }

  // Lights up one risk, marks earlier ones as done, and swaps the panel on the right.
  const focus = (risk: number) => {
    timeline.to(risk ? `.panel-${risk - 1}` : '.empty', { opacity: 0, ...exit })
    risks.forEach((_, index) => style(index, index < risk ? 'done' : index === risk ? 'on' : 'off', '<'))
    timeline.to(`.panel-${risk}`, { opacity: 1, ...enter }, '<0.1')
  }

  risks.forEach((_, index) => {
    const look = looks.neutral
    gsap.set(`.item-${index} .name`, { color: colour(look.name) })
    gsap.set(`.item-${index} .badge`, { borderColor: colour(look.ring), backgroundColor: colour(look.fill) })
    gsap.set(`.item-${index} .number`, { color: colour(look.number) })
  })
  gsap.set('.tick, .plan, .panel, .reveal, .traveller, .whole-check', { opacity: 0 })
  gsap.set('.check', { scale: 0.4, transformOrigin: '50% 50%' })
  gsap.set('.memory-bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.sample-line', { scaleX: 0, transformOrigin: '0% 50%' })

  timeline.addLabel('step0')

  // Click 1. Each part of the pipeline is tested on its own.
  focus(0)
  timeline
    .to('.panel-0 .tests', { opacity: 1, ...enter, stagger: 0.12 }, '>')
    .to('.panel-0 .tests .check', { opacity: 1, scale: 1, ...pop, stagger: 0.12 }, '<0.1')
    .to('.panel-0 .tested-note', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step1')

  // Click 2. A small sample of posts runs through the whole pipeline.
  timeline
    .to('.panel-0 .tests, .panel-0 .tested-note', { opacity: faded, ...exit })
    .to('.panel-0 .links, .panel-0 .sample', { opacity: 1, ...enter }, '<0.1')
    .to('.sample-line', { opacity: 1, duration: 0 }, '>')
    .to('.sample-line', { scaleX: 1, ...draw }, '<')
    .to('.traveller', { opacity: 1, duration: 0 }, '<0.2')
    .fromTo('.traveller', { x: 0 }, { x: sample.to - sample.from - 34, ...moveShort, immediateRender: false }, '<')
    .to('.traveller', { opacity: 0, ...exit }, '>')
    .to('.panel-0 .whole-check', { opacity: 1, scale: 1, ...pop }, '<')
    .addLabel('step2')

  // Click 3. Four optimizers need more than 8 GB.
  focus(1)
  timeline
    .to('.memory-bar', { scaleY: 1, ...draw, stagger: 0.03 }, '<0.1')
    .to('.panel-1 .over-note', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step3')

  // Click 4. The embedding layer is frozen, and every bar drops below the line.
  timeline
    .to('.panel-1 .over-note', { opacity: 0, ...exit })
    .to('.panel-1 .before', { opacity: 1, ...enter }, '<')
    .to('.memory-bar', { scaleY: chart.frozen, ...draw }, '<0.1')
    .to('.panel-1 .model', { opacity: 1, ...enter }, '<')
    .to('.panel-1 .fit-note', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step4')

  // Click 5. A failed use case is fixed and tested again until it passes.
  focus(2)
  timeline
    .to('.panel-2 .loop', { opacity: 1, ...enter, stagger: 0.18 }, '>')
    .to('.panel-2 .passes', { opacity: 1, ...enter }, '>0.1')
    .addLabel('step5')

  // Click 6. X API falls back to public Facebook posts.
  focus(3)
  timeline
    .to('.panel-3 .facebook', { opacity: 1, ...enter }, '>')
    .addLabel('step6')

  // Click 7. If posts are still short, the smaller dataset is reported.
  timeline
    .to('.panel-3 .first', { opacity: faded, ...exit })
    .to('.panel-3 .limitation', { opacity: 1, ...enter }, '<0.1')
    .addLabel('step7')

  // Click 8. A delay slides into the buffer month, and the chain returns to full strength.
  timeline
    .to('.panel-3 .limitation', { opacity: faded, ...exit })
    .to('.panel-3 .delays', { opacity: 1, ...enter }, '<0.1')
    .fromTo('.delay', { x: -70, y: -46 }, { x: 0, y: 0, ...moveShort, immediateRender: false }, '>')
    .to('.panel-3 .first, .panel-3 .limitation', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step8')
})
</script>

<template>
  <div ref="root" class="risks">
    <section>
      <h3>Four risks</h3>
      <ol>
        <li v-for="(risk, index) in risks" :key="risk.name" class="item" :class="`item-${index}`">
          <span class="badge">
            <span class="number">{{ index + 1 }}</span>
            <svg class="tick" viewBox="-15 -15 30 30" width="30" height="30"><path d="M -7 0 L -2 5 L 7 -5" /></svg>
          </span>
          <div>
            <p class="name">{{ risk.name }}</p>
            <p class="plan">{{ risk.plan }}</p>
          </div>
        </li>
      </ol>
    </section>

    <svg viewBox="0 0 440 456" width="440" height="456">
      <defs>
        <marker id="risks-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
          <path class="head" d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
        <marker id="risks-head-data" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto">
          <path class="head data" d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
        <marker id="risks-head-pass" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto">
          <path class="head pass" d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
      </defs>

      <text class="empty" x="210" y="230" text-anchor="middle">Each risk has a backup plan</text>

      <!-- Risk 1. Code bugs -->
      <g class="panel panel-0">
        <g v-for="(part, index) in parts" :key="part.name">
          <rect class="code-box" :x="part.x" y="90" width="92" height="60" rx="8" />
          <text class="box-text" :x="part.x + 46" :y="index === 2 ? 116 : 125" text-anchor="middle">{{ part.name }}</text>
          <text v-if="index === 2" class="code-text" :x="part.x + 46" y="136" text-anchor="middle">all 13</text>
        </g>
        <g v-for="(part, index) in parts" :key="`test-${part.name}`" class="reveal tests">
          <path class="test-link" :d="`M ${part.x + 46} 152 V 192`" />
          <rect class="test-pill" :x="part.x + 4" y="194" width="84" height="28" rx="14" />
          <text class="test-text" :x="part.x + 36" y="213" text-anchor="middle">{{ index === 2 ? 'Each one' : 'Tested' }}</text>
          <g class="check" :transform="`translate(${part.x + 74} 208)`">
            <circle r="8" />
            <path d="M -3.6 0 L -1.2 3 L 3.6 -3" />
          </g>
        </g>
        <text class="reveal tested-note small" x="0" y="256">Small examples with known answers</text>

        <g class="reveal links">
          <path v-for="part in parts.slice(0, 3)" :key="`link-${part.name}`" class="link" :d="`M ${part.x + 94} 120 H ${part.x + 108}`" marker-end="url(#risks-head)" />
        </g>
        <g class="reveal sample">
          <rect v-for="layer in 3" :key="layer" class="post" :x="4 + (layer - 1) * 7" :y="sample.y - 8 - (layer - 1) * 7" width="44" height="32" rx="5" />
          <text class="data-text" x="0" :y="sample.y - 36">Small sample of posts</text>
          <text class="data-text" :x="(sample.from + sample.to) / 2" :y="sample.y - 12" text-anchor="middle">Runs through the whole pipeline</text>
        </g>
        <path class="sample-line" :d="`M ${sample.from} ${sample.y} H ${sample.to}`" marker-end="url(#risks-head-data)" />
        <rect class="traveller post" :x="sample.from" :y="sample.y - 8" width="22" height="16" rx="3" />
        <g class="check whole-check" :transform="`translate(410 ${sample.y})`">
          <circle r="12" />
          <path d="M -5.4 0 L -1.8 4.5 L 5.4 -4.5" />
        </g>
      </g>

      <!-- Risk 2. Hardware limits -->
      <g class="panel panel-1">
        <path class="chart-axis" :d="`M ${chart.left - 8} 100 V ${chart.base} H ${chart.right + 6}`" />
        <text class="small" text-anchor="middle" :transform="`translate(${chart.left - 20} 290) rotate(-90)`">Memory needed</text>
        <text class="small" :x="(chart.left + chart.right) / 2" :y="chart.base + 22" text-anchor="middle">13 optimizers</text>
        <rect v-for="(bar, index) in bars" :key="`before-${index}`" class="reveal before" :x="bar.x" :y="chart.base - bar.height" :width="bar.width" :height="bar.height" />
        <rect v-for="(bar, index) in bars" :key="index" class="memory-bar" :x="bar.x" :y="chart.base - bar.height" :width="bar.width" :height="bar.height" />
        <path class="limit" :d="`M ${chart.left - 8} ${limit} H ${chart.right + 6}`" />
        <text class="limit-text" :x="chart.left - 14" :y="limit + 5" text-anchor="end">8 GB</text>
        <text class="reveal over-note note" :x="chart.right" y="98" text-anchor="end">Four go over the 8 GB limit</text>
        <text class="reveal fit-note note" :x="chart.right" y="98" text-anchor="end">All fit after freezing</text>

        <g class="reveal model">
          <text class="small" :x="stack.x" y="126">Each model</text>
          <rect class="layer" :x="stack.x" y="136" :width="stack.width" height="34" rx="5" />
          <text class="layer-text" :x="stack.x + stack.width / 2" y="158" text-anchor="middle">Task heads</text>
          <rect class="layer" :x="stack.x" y="174" :width="stack.width" height="62" rx="5" />
          <text class="layer-text" :x="stack.x + stack.width / 2" y="210" text-anchor="middle">Encoder layers</text>
          <rect class="layer frozen" :x="stack.x" y="240" :width="stack.width" height="34" rx="5" />
          <text class="layer-text frozen-text" :x="stack.x + stack.width / 2 - 10" y="262" text-anchor="middle">Embedding</text>
          <g class="padlock" :transform="`translate(${stack.x + stack.width - 18} 257)`">
            <path d="M -3.5 -2 v -4 a 3.5 3.5 0 0 1 7 0 v 4" />
            <rect x="-6.5" y="-2" width="13" height="10" rx="2" />
          </g>
          <text class="frozen-name" :x="stack.x" y="298">Frozen</text>
        </g>
      </g>

      <!-- Risk 3. Prototype failures -->
      <g class="panel panel-2">
        <text class="small" x="10" y="70">Prototype web application, each use case</text>
        <g class="reveal loop">
          <rect class="test-box" x="15" y="100" width="150" height="54" rx="10" />
          <text class="step-text" x="90" y="132" text-anchor="middle">Test use case</text>
        </g>
        <g class="reveal loop">
          <path class="link" d="M 168 127 H 264" marker-end="url(#risks-head)" />
          <rect class="fail-box" x="270" y="100" width="150" height="54" rx="10" />
          <text class="step-text failed" x="345" y="132" text-anchor="middle">Fails</text>
        </g>
        <g class="reveal loop">
          <path class="link" d="M 345 156 Q 345 277 301 277" marker-end="url(#risks-head)" />
          <rect class="code-box" x="145" y="250" width="150" height="54" rx="10" />
          <text class="step-text" x="220" y="282" text-anchor="middle">Fix it</text>
        </g>
        <g class="reveal loop">
          <path class="link" d="M 145 277 Q 110 277 110 162" marker-end="url(#risks-head)" />
          <text class="again" x="220" y="205" text-anchor="middle">Test it again</text>
        </g>
        <g class="reveal passes">
          <path class="pass-line" d="M 50 156 V 340" marker-end="url(#risks-head-pass)" />
          <rect class="pass-box" x="0" y="346" width="110" height="44" rx="10" />
          <text class="step-text pass-text" x="55" y="373" text-anchor="middle">Passes</text>
        </g>
      </g>

      <!-- Risk 4. Data and delays -->
      <g class="panel panel-3">
        <g class="first">
          <rect class="source" x="10" y="55" width="220" height="46" rx="10" />
          <text class="step-text" x="120" y="84" text-anchor="middle">Posts from X API</text>
        </g>
        <g class="reveal facebook first">
          <path class="link" d="M 120 103 V 145" marker-end="url(#risks-head)" />
          <text class="condition" x="136" y="129">If it fails or gives too few Taglish posts</text>
          <rect class="source" x="10" y="149" width="220" height="46" rx="10" />
          <text class="step-text" x="120" y="178" text-anchor="middle">Public Facebook posts</text>
        </g>
        <g class="reveal limitation">
          <path class="link" d="M 120 197 V 239" marker-end="url(#risks-head)" />
          <text class="condition" x="136" y="223">If still short</text>
          <rect class="limit-box" x="10" y="243" width="220" height="58" rx="10" />
          <text class="step-text" x="120" y="268" text-anchor="middle">Report smaller dataset</text>
          <text class="small" x="120" y="287" text-anchor="middle">as a limitation of the study</text>
        </g>
        <g class="reveal delays">
          <text class="small" x="10" y="340">Any other delay</text>
          <rect class="planned" x="10" y="352" width="290" height="28" rx="5" />
          <text class="layer-text" x="155" y="371" text-anchor="middle">Planned work</text>
          <rect class="buffer" x="304" y="352" width="126" height="28" rx="5" />
          <text class="small" x="367" y="400" text-anchor="middle">Buffer month</text>
          <path class="link" d="M 280 342 Q 310 328 326 352" marker-end="url(#risks-head)" />
          <g class="delay">
            <rect class="delay-box" x="310" y="357" width="54" height="18" rx="3" />
            <text class="delay-text" x="337" y="370" text-anchor="middle">Delay</text>
          </g>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.risks {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 456px;
}

section {
  flex: none;
  width: 390px;
}

h3 {
  margin-bottom: var(--space-3);
}

ol {
  display: grid;
  gap: var(--space-3);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: grid;
  grid-template-columns: 2.75rem 1fr;
  align-items: start;
  line-height: var(--leading-list);
}

.badge {
  position: relative;
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border: 1.5px solid;
  border-radius: 50%;
}

.number {
  font-size: 0.95rem;
  font-weight: 700;
}

.tick {
  position: absolute;
  inset: -1.5px;
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.name {
  margin: 0;
  font-weight: 700;
  line-height: var(--leading-list);
}

.plan {
  margin: 2px 0 0;
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
}

svg {
  flex: none;
  overflow: visible;
}

.empty {
  font-size: 1rem;
  fill: var(--color-faint);
}

.head {
  fill: var(--color-subtle);
}

.head.data {
  fill: var(--color-data);
}

.head.pass {
  fill: var(--color-positive);
}

.link {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.small {
  font-size: 0.85rem;
  fill: var(--color-subtle);
}

.note {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-muted);
}

.box-text,
.step-text {
  font-size: 15px;
  font-weight: 600;
  fill: var(--color-ink);
}

.step-text {
  font-size: 16px;
}

.code-box {
  fill: var(--color-surface);
  stroke: var(--color-coding);
  stroke-width: 2;
}

.code-text,
.test-text {
  font-size: 0.8rem;
  fill: var(--color-coding);
}

.test-link {
  stroke: var(--color-testing);
  stroke-width: 1.5;
  stroke-dasharray: 3 3;
}

.test-pill,
.test-box {
  fill: var(--color-testing-tint);
  stroke: var(--color-testing);
  stroke-width: 1.5;
}

.test-box {
  stroke-width: 2;
}

.check circle {
  fill: var(--color-positive);
}

.check path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.post {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
  stroke-width: 1.5;
}

.data-text {
  font-size: 0.85rem;
  fill: var(--color-data);
}

.sample-line {
  fill: none;
  stroke: var(--color-data);
  stroke-width: 2;
  stroke-dasharray: 7 5;
}

.chart-axis {
  fill: none;
  stroke: var(--color-faint);
}

.memory-bar {
  fill: var(--color-ink);
}

.before {
  fill: none;
  stroke: var(--color-removed);
  stroke-dasharray: 3 3;
}

.limit {
  stroke: var(--color-ink);
  stroke-width: 1.5;
  stroke-dasharray: 6 4;
}

.limit-text {
  font-size: 0.85rem;
  font-weight: 600;
  fill: var(--color-ink);
}

.layer {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 1.5;
}

.layer.frozen {
  fill: var(--color-land);
  stroke: var(--color-subtle);
}

.layer-text {
  font-size: 0.8rem;
  fill: var(--color-ink);
}

.frozen-text {
  fill: var(--color-muted);
}

.padlock path {
  fill: none;
  stroke: var(--color-subtle);
  stroke-width: 1.8;
}

.padlock rect {
  fill: var(--color-subtle);
}

.frozen-name {
  font-size: 0.85rem;
  font-weight: 700;
  fill: var(--color-muted);
}

.fail-box {
  fill: var(--color-land);
  stroke: var(--color-removed);
  stroke-width: 1.5;
}

.failed {
  font-weight: 500;
  fill: var(--color-subtle);
}

.again,
.condition {
  font-size: 0.85rem;
  font-style: italic;
  fill: var(--color-subtle);
}

.pass-line {
  fill: none;
  stroke: var(--color-positive);
  stroke-width: 2;
}

.pass-box {
  fill: var(--color-surface);
  stroke: var(--color-positive);
  stroke-width: 2;
}

.pass-text {
  fill: var(--color-positive);
}

.source {
  fill: var(--color-data-tint);
  stroke: var(--color-data);
  stroke-width: 2;
}

.limit-box {
  fill: var(--color-surface);
  stroke: var(--color-subtle);
  stroke-width: 2;
}

.planned {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 1.5;
}

.buffer {
  fill: none;
  stroke: var(--color-removed);
  stroke-width: 1.5;
  stroke-dasharray: 5 4;
}

.delay-box {
  fill: var(--color-land);
  stroke: var(--color-subtle);
}

.delay-text {
  font-size: 0.7rem;
  fill: var(--color-muted);
}
</style>
