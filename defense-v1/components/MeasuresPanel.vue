<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const pop = { duration: 0.25, ease: 'back.out(3)' }

// One made-up Taglish post on three lines. Each piece is a word or phrase. The answer key marks
// the three real details, and the model marks two of them and one word that is not a detail
const lines = [['Baha', 'sa', 'Marikina,'], ['need ng tubig', 'para', 'sa'], ['mga bata!']]
const real = new Set(['Marikina,', 'need ng tubig', 'mga bata!'])
const answers = new Set(['Baha', 'Marikina,', 'need ng tubig'])

// Every number is the model's score, comparing its answers with the key
const correct = [...answers].filter(piece => real.has(piece)).length
const precision = correct / answers.size
const recall = correct / real.size
const f1 = (2 * precision * recall / (precision + recall)).toFixed(2).replace(/^0/, '')

// The side not being discussed fades back but stays readable
const dim = 0.35

// Illustration only. How much the multi-task model lost, as a share of the bar and as a score
const lost = 14
const loss = '−0.03'

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.model-side, .count-row, .check, .not-found, .marks-legend, .scene.transfer, .scene.cost, .row, .transfer-part', { opacity: 0 })
  gsap.set('.check', { scale: 0.4, transformOrigin: '50% 50%' })
  // The multi-task bar starts as long as the single-task bar, then falls short of it
  gsap.set('.multi', { width: '100%' })
  gsap.set('.drop, .dot-value', { opacity: 0 })
  gsap.set('.dot-wrap', { y: 0 })
  gsap.set('.fall', { scaleY: 0, transformOrigin: '50% 0%' })

  timeline
    .addLabel('step0')
    // The model's answers on the right, and how many of them are correct
    .to('.key-side', { opacity: dim, ...exit })
    .to('.model-side', { opacity: 1, ...enter }, '<')
    .to('.count-row.correct, .marks-legend', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step1')
    // Back on the key, which stays correct, the details the model found are checked.
    // Then how many it found, and the F1-score that balances the two
    .to('.model-side .answers', { opacity: dim, ...exit })
    .to('.key-side', { opacity: 1, ...enter }, '<')
    .to('.check', { opacity: 1, scale: 1, ...pop, stagger: 0.15 }, '>')
    .to('.not-found', { opacity: 1, ...enter }, '>0.1')
    .to('.count-row.found', { opacity: 1, ...enter }, '>0.1')
    .to('.count-row.f1', { opacity: 1, ...enter }, '>0.3')
    .addLabel('step2')
    // Negative transfer
    .to('.scene.f1', { opacity: 0, ...exit })
    .to('.scene.transfer', { opacity: 1, ...enter })
    .to('.transfer-part.bars', { opacity: 1, ...enter }, '<0.1')
    .to('.multi', { width: `${100 - lost}%`, ...draw })
    .to('.drop', { opacity: 1, ...enter }, '<0.2')
    // The same loss on the axis. The dot drops from 0 to below it
    .to('.transfer-part.axis', { opacity: 1, ...enter }, '<')
    .to('.dot-wrap', { y: 38, ...draw }, '<0.2')
    .to('.fall', { scaleY: 1, ...draw }, '<')
    .to('.dot-value', { opacity: 1, ...enter }, '>-0.1')
    .to('.transfer-part.result', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step3')
    // Speed
    .to('.scene.transfer, .footer', { opacity: 0, ...exit })
    .to('.scene.cost', { opacity: 1, ...enter })
    .to('.row.speed', { opacity: 1, ...enter }, '<')
    .addLabel('step4')
    // Memory
    .to('.row.memory', { opacity: 1, ...enter })
    .addLabel('step5')
})
</script>

<template>
  <div ref="root" class="measures">
    <section class="scene f1">
      <div class="tag">Accuracy, the F1-score</div>
      <div class="sides">
        <div class="side key-side">
          <div class="tag">Answer key</div>
          <div class="post-card">
            <p v-for="(line, row) in lines" :key="row" class="post-line">
              <span v-for="piece in line" :key="piece" class="piece">
                <span v-if="real.has(piece)" class="box right" />
                <svg v-if="real.has(piece) && answers.has(piece)" class="check" viewBox="-11 -11 22 22" width="22" height="22">
                  <circle r="10" />
                  <path d="M -4.5 0 L -1 3.5 L 5 -4" />
                </svg>
                {{ piece }}
              </span>
              <span v-if="line.some(piece => real.has(piece) && !answers.has(piece))" class="not-found">Not found by the model</span>
            </p>
          </div>
        </div>

        <div class="side model-side">
          <div class="answers">
            <div class="tag">Model's answers</div>
            <div class="post-card">
              <p v-for="(line, row) in lines" :key="row" class="post-line">
                <span v-for="piece in line" :key="piece" class="piece">
                  <span v-if="answers.has(piece)" class="box" :class="real.has(piece) ? 'right' : 'wrong'" />
                  {{ piece }}
                </span>
              </p>
            </div>
          </div>
          <div class="counts">
            <div class="count-row correct"><b>{{ correct }} of {{ answers.size }}</b><span>of its answers are correct</span></div>
            <div class="count-row found"><b>{{ correct }} of {{ real.size }}</b><span>of the correct answers found</span></div>
            <div class="count-row f1"><b>{{ f1 }}</b><span>F1-score, balancing the two</span></div>
          </div>
        </div>
      </div>
      <div class="marks-legend"><span><i class="key right" />Correct</span><span><i class="key wrong" />Wrong</span></div>
    </section>

    <section class="scene transfer">
      <div class="tag">Accuracy, negative transfer</div>
      <p class="statement">Negative transfer is the multi-task score minus the single-task score</p>
      <div class="transfer-layout">
        <div class="transfer-part bars">
          <div class="pair">
            <span class="who-label">Single-task model</span>
            <div class="long single" />
          </div>
          <div class="pair">
            <span class="who-label">Multi-task model</span>
            <div class="long track">
              <div class="multi" />
              <!-- The part of the single-task score that the multi-task model lost -->
              <div class="drop" :style="{ width: `${lost}%` }"><span class="drop-label">Lost, {{ loss }}</span></div>
            </div>
          </div>
        </div>
        <div class="transfer-part axis">
          <span class="axis-label gain">Above 0, a gain</span>
          <div class="axis-line" />
          <div class="zero"><span>0</span></div>
          <div class="fall" />
          <div class="dot-wrap">
            <div class="dot" />
            <span class="dot-value">{{ loss }}</span>
          </div>
          <span class="axis-label loss">Below 0, a loss</span>
        </div>
      </div>
      <p class="transfer-part result">Below 0, the multi-task model lost accuracy, so this is negative transfer</p>
    </section>

    <section class="scene cost">
      <div v-for="row in [
        { key: 'speed', name: 'Speed', cards: [{ icon: 'clock', title: 'Training time' }, { icon: 'bolt', title: 'Time to label each post' }] },
        { key: 'memory', name: 'Memory', cards: [{ icon: 'chip', title: 'Peak memory during training' }, { icon: 'disk', title: 'Size of the saved model' }] },
      ]" :key="row.key" class="row" :class="row.key"
      >
        <div class="tag">{{ row.name }}</div>
        <div class="cards">
          <div v-for="card in row.cards" :key="card.title" class="card">
            <svg viewBox="-16 -16 32 32" width="36" height="36" class="icon">
              <template v-if="card.icon === 'clock'">
                <circle r="12" />
                <path d="M 0 -6 V 0 L 5 4" />
              </template>
              <path v-else-if="card.icon === 'bolt'" d="M 3 -13 L -8 3 H 0 L -3 13 L 8 -3 H 0 Z" />
              <template v-else-if="card.icon === 'chip'">
                <rect x="-9" y="-9" width="18" height="18" rx="2" />
                <path d="M -4 -9 V -14 M 4 -9 V -14 M -4 9 V 14 M 4 9 V 14 M -9 -4 H -14 M -9 4 H -14 M 9 -4 H 14 M 9 4 H 14" />
              </template>
              <template v-else>
                <path d="M -11 -12 H 7 L 11 -8 V 12 H -11 Z" />
                <path d="M -6 -12 V -5 H 5 V -12 M -6 12 V 3 H 6 V 12" />
              </template>
            </svg>
            <span class="card-title">{{ card.title }}</span>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">Illustration, made-up posts and scores</div>
  </div>
</template>

<style scoped>
.measures {
  position: relative;
  height: 456px;
}

.scene {
  position: absolute;
  inset: 0;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-1);
}

/* The answer key on the left, the model's answers and its score on the right */
.sides {
  display: grid;
  grid-template-columns: 380px 380px;
  justify-content: space-between;
  margin-top: var(--space-2);
}

.post-card {
  padding: 14px 20px;
  border: 2px solid var(--color-faint);
  border-radius: 12px;
}

.post-line {
  margin: 0;
  font-size: var(--text-sm);
  line-height: 2.3;
}

.piece {
  position: relative;
  /* Keeps each box behind its word but in front of the slide */
  isolation: isolate;
  display: inline-block;
  padding: 0 7px;
  margin-right: 2px;
  line-height: 1.6;
}

.box {
  position: absolute;
  inset: 0;
  border-radius: 5px;
  z-index: -1;
}

.box.right,
.key.right {
  background: color-mix(in srgb, var(--color-ner) 22%, transparent);
  border: 2px solid var(--color-ner);
}

.box.wrong,
.key.wrong {
  background: color-mix(in srgb, var(--color-negative) 14%, transparent);
  border: 2px solid var(--color-negative);
}

/* A detail of the key that the model also marked */
.check {
  position: absolute;
  right: -11px;
  top: -11px;
}

.check circle {
  fill: #15803d;
}

.check path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.not-found {
  margin-left: var(--space-1);
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-subtle);
}

.counts {
  margin-top: var(--space-2);
}

.count-row {
  display: grid;
  grid-template-columns: 112px 1fr;
  column-gap: var(--space-1);
  align-items: baseline;
  padding: 2px 0;
}

.count-row b {
  font-size: var(--text-3xl);
  font-weight: 700;
  letter-spacing: -0.02em;
  text-align: right;
}

.count-row span {
  font-size: var(--text-2xs);
  font-weight: 600;
}

.count-row.f1 {
  margin-top: var(--space-1);
  padding-top: var(--space-1);
  border-top: 2px solid var(--color-land);
}

.count-row.f1 span {
  color: var(--color-muted);
}

.marks-legend {
  position: absolute;
  left: 0;
  bottom: 0;
  display: flex;
  gap: var(--space-3);
  font-size: var(--text-2xs);
  color: var(--color-muted);
}

.key {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 3px;
  margin-right: 8px;
  vertical-align: -2px;
}

.statement {
  font-size: var(--text-md);
  font-weight: 600;
  margin: var(--space-1) 0 var(--space-4);
}

.transfer-layout {
  display: grid;
  grid-template-columns: 1fr 200px;
  column-gap: var(--space-5);
  align-items: center;
}

.pair {
  display: grid;
  grid-template-columns: 190px 1fr;
  align-items: center;
  margin-bottom: var(--space-2);
  font-size: var(--text-xs);
}

.long {
  height: 28px;
  border-radius: 4px;
}

.single {
  width: 100%;
  background: var(--color-subtle);
}

.track {
  position: relative;
  width: 100%;
}

.multi {
  height: 100%;
  border-radius: 4px;
  background: var(--color-ink);
}

/* An empty, dashed outline where the lost score would have been */
.drop {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  box-sizing: border-box;
  border: 2px dashed var(--color-negative);
  border-radius: 4px;
  background: color-mix(in srgb, var(--color-negative) 8%, var(--color-surface));
}

.drop-label {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  white-space: nowrap;
  font-size: var(--text-2xs);
  font-weight: 700;
  color: var(--color-negative);
}

.axis {
  position: relative;
  height: 160px;
}

.axis-line {
  position: absolute;
  left: 50%;
  top: 28px;
  bottom: 28px;
  border-left: 2px solid var(--color-ink);
}

.zero {
  position: absolute;
  left: calc(50% - 30px);
  top: 50%;
  width: 60px;
  border-top: 2px solid var(--color-ink);
}

.zero span {
  position: absolute;
  right: -20px;
  top: -12px;
  font-size: var(--text-2xs);
  font-weight: 600;
}

/* The red line the dot leaves as it drops below 0 */
.fall {
  position: absolute;
  left: calc(50% - 2px);
  top: 50%;
  width: 4px;
  height: 38px;
  background: var(--color-negative);
}

.dot-wrap {
  position: absolute;
  left: calc(50% - 8px);
  top: calc(50% - 8px);
}

.dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-negative);
}

.dot-value {
  position: absolute;
  left: 24px;
  top: -4px;
  white-space: nowrap;
  font-size: var(--text-xs);
  font-weight: 700;
  color: var(--color-negative);
}

.axis-label {
  position: absolute;
  left: 0;
  right: 0;
  text-align: center;
  font-size: var(--text-2xs);
  color: var(--color-muted);
}

.axis-label.gain {
  top: 0;
}

.axis-label.loss {
  bottom: 0;
  color: var(--color-negative);
}

.transfer-part.result {
  margin-top: var(--space-3);
  font-size: var(--text-md);
  font-weight: 700;
  color: var(--color-negative);
}

.row {
  margin-bottom: var(--space-4);
}

.cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-2);
}

.card {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: 10px;
  background: var(--color-land);
}

.card-title {
  font-size: var(--text-base);
  font-weight: 600;
}

.icon {
  flex: none;
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.footer {
  position: absolute;
  right: 0;
  top: 0;
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
