<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, moveLong, pop } from '../shared/motion'

// Each marked phrase is correct (c), wrong (w) or missed by the model (m).
type Kind = 'c' | 'w' | 'm'
type Token = string | [string, Kind]
const posts: Token[][] = [
  [['Baha', 'w'], ' sa ', ['Marikina', 'c'], ', ', ['need ng tubig', 'm'], ' para sa mga bata!'],
  [['Walang kuryente', 'c'], ' dito sa ', ['Tacloban', 'c'], ' since kagabi'],
  ['Stranded kami sa ', ['Brgy. San Roque', 'c'], ', ', ['pls rescue', 'm']],
  ['Ang lakas ng ', ['hangin', 'w'], ' sa ', ['Albay', 'c'], ' grabe'],
  [['Need ng pagkain', 'c'], ' sa ', ['evacuation center', 'w'], ' sa ', ['Naga', 'm']],
  ['Sira ang ', ['tulay', 'm'], ' sa ', ['Cagayan', 'c'], ', di makadaan'],
  ['Praying for everyone sa ', ['Bicol', 'c']],
  ['Lubog na bahay namin sa ', ['Malabon', 'c'], ', ', ['help po', 'm']],
  ['Nag-donate kami ng ', ['relief goods', 'w'], ' sa ', ['Iloilo', 'c']],
  ['Ilang araw ', ['walang signal', 'c'], ' sa ', ['Catanduanes', 'm']],
  [['Need ng gamot', 'c'], ' para kay lola sa ', ['Quezon City', 'c']],
  ['Ingat kayo sa ', ['Pampanga', 'c'], ' mga ', ['besh', 'w']],
  ['Bumabaha na sa ', ['Bulacan', 'c'], ', ', ['taas ng tubig', 'm']],
  [['Nawalan ng bahay', 'c'], ' ang ', ['pamilya', 'w'], ' sa ', ['Samar', 'c']],
  ['Sarado ang ', ['NLEX', 'c'], ' dahil sa ', ['landslide', 'm']],
  ['Grabe ang ', ['ulan', 'w'], ' sa ', ['Batangas', 'c'], ' ngayon'],
]
const sups: Record<Kind, string> = { c: '✓', w: '✕', m: 'missed' }

// The first post, shown large. Every piece is its own span, so the words always stay in order.
const firstPost = posts[0].map(token => typeof token === 'string' ? { text: token } : { text: token[0], kind: token[1] })

type Mark = 'ok' | 'no' | 'ms'
type Piece = { text: string, mark?: Mark }
const modes = ['judged', 'eager', 'careful', 'balanced'] as const
type Mode = typeof modes[number]

// The eager model also marks most ordinary words, so short and common words stay plain.
const common = /^(?:sa|ng|na|ang|mga|para|kay|di|po|for|since|kami|kayo|dito|namin)$/i
function render(post: Token[], mode: Mode, postIndex: number): Piece[] {
  let found = 0
  return post.flatMap((token): Piece[] => {
    if (typeof token === 'string') {
      if (mode !== 'eager')
        return [{ text: token }]
      return token.split(/([A-Za-z!-]+)/).filter(Boolean)
        .map(word => /^[A-Za-z!-]+$/.test(word) && word.length >= 3 && !common.test(word) ? { text: word, mark: 'no' } : { text: word })
    }
    const [text, kind] = token
    if (mode === 'judged')
      return [{ text, mark: kind === 'c' ? 'ok' : kind === 'w' ? 'no' : 'ms' }]
    if (kind === 'w')
      return [mode === 'eager' ? { text, mark: 'no' } : { text }]
    const index = found++
    if (mode === 'eager')
      return [{ text, mark: 'ok' }]
    if (mode === 'careful')
      return [{ text, mark: index === 0 && postIndex % 2 === 0 ? 'ok' : 'ms' }]
    return [{ text, mark: (postIndex === 2 || postIndex === 7) && index === 1 ? 'ms' : 'ok' }]
  })
}

// The training posts from the split, 72 blocks in 9 columns.
const blockSize = 30
const blockStep = 36
const blockColumns = 9
const blockOrigin = { x: (884 - (blockColumns * blockStep - 6)) / 2, y: (456 - (8 * blockStep - 6)) / 2 }
const lifted = 31
const blockAt = (index: number) => ({
  left: blockOrigin.x + (index % blockColumns) * blockStep,
  top: blockOrigin.y + Math.floor(index / blockColumns) * blockStep,
})
const blocks = Array.from({ length: 72 }, (_, index) => index).filter(index => index !== lifted).map(blockAt)

const bigCard = { left: 0, top: 56, width: 884, height: 300 }
const cellAt = (index: number) => ({
  left: (index % 4) * 148,
  top: 15 + Math.floor(index / 4) * 110,
  width: 134,
  height: 96,
})

// In the last click, three groups of four cards sit side by side.
const groupWidth = 270
const groupStep = 307
const trioAt = (index: number) => {
  const group = Math.floor(index / 4)
  const place = index % 4
  return {
    left: group * groupStep + (place % 2) * 140,
    top: Math.floor(place / 2) * 102,
    width: 130,
    height: 92,
  }
}
const trio = [
  { what: 'Many wrong answers', verdict: 'Low' },
  { what: 'Many missed answers', verdict: 'Low' },
  { what: 'Few of either', verdict: 'High' },
].map((label, group) => ({ ...label, left: group * groupStep }))

const cards = Array.from({ length: 16 }, (_, index) => ({
  group: index < 12 ? `group-${Math.floor(index / 4)}` : 'leaving',
  layers: modes.map(mode => ({ mode, pieces: render(posts[index % posts.length], mode, index % posts.length) })),
}))

const px = (box: Record<string, number>) => Object.fromEntries(Object.entries(box).map(([key, value]) => [key, `${value}px`]))

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()
  const local = (name: string) => css.getPropertyValue(`--${name}`).trim()

  gsap.set('.eyebrow, .big-text, .legend, .card:not(.first), .first .layer, .layer:not(.judged), .panel > *, .trio-label, .outline, .rule', { opacity: 0 })
  gsap.set('.first', { ...blockAt(lifted), width: blockSize, height: blockSize, borderRadius: 4, backgroundColor: colour('data') })
  gsap.set('.sup', { opacity: 0, scale: 0.4 })

  timeline.addLabel('step0')

  // Click 1. One block lifts out of the group and grows into a post.
  timeline
    .to('.block', { opacity: 0, ...exit, stagger: { amount: 0.2, from: 'random' } })
    .to('.first', { scale: 1.3, duration: 0.25, ease: 'power2.out' }, '<')
    .to('.first', { ...bigCard, borderRadius: 14, scale: 1, backgroundColor: colour('surface'), ...moveLong })
    .to('.eyebrow.one, .big-text', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step1')

  // Click 2. The model marks its answers one at a time.
    .to('.eyebrow.one', { opacity: 0, ...exit })
    .to('.eyebrow.model', { opacity: 1, ...enter }, '<')
    .fromTo('.big-text .k.c, .big-text .k.w', { scale: 0.85 }, { scale: 1, backgroundColor: local('answer-fill'), borderColor: colour('ner'), color: local('answer-ink'), ...pop, stagger: 0.3 }, '<0.1')
    .addLabel('step2')

  // Click 3. The answer key shows which marks are correct, wrong or missed.
    .to('.eyebrow.model', { opacity: 0, ...exit })
    .to('.eyebrow.judged', { opacity: 1, ...enter }, '<')
    .to('.big-text .k.c', { backgroundColor: local('correct-fill'), borderColor: local('correct-line'), color: local('correct-ink'), ...enter }, '<0.1')
    .to('.big-text .k.c .sup', { opacity: 1, scale: 1, ...pop }, '<')
    .to('.big-text .k.w', { backgroundColor: colour('land'), borderColor: colour('subtle'), color: colour('muted'), textDecorationColor: colour('muted'), ...enter }, '>0.1')
    .to('.big-text .k.w .sup', { opacity: 1, scale: 1, ...pop }, '<')
    .to('.big-text .k.m', { borderColor: colour('subtle'), ...enter }, '>0.1')
    .to('.big-text .k.m .sup', { opacity: 1, scale: 1, ...pop }, '<')
    .to('.legend', { opacity: 1, ...enter }, '>0.1')
    .addLabel('step3')

  // Click 4. The view pulls back to show the same check on many posts.
    .to('.eyebrow, .legend, .big-text', { opacity: 0, ...exit })
    .to('.first', { ...cellAt(0), borderRadius: 10, ...moveLong }, '<0.1')
    .to('.first .layer.judged', { opacity: 1, ...enter }, '>-0.2')
    .fromTo('.card:not(.first)', { scale: 0.9 }, { opacity: 1, scale: 1, ...enter, stagger: 0.05 }, '<')
    .to('.panel.every > *', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step4')

  // Clicks 5 and 6. The same posts are labeled by an eager model, then by a careful one.
  const relabel = (from: Mode, to: Mode, panel: string, previous: string) => timeline
    .to(`.panel.${previous} > *`, { opacity: 0, ...exit })
    .to(`.layer.${from}`, { opacity: 0, ...exit }, '<')
    .to(`.layer.${to}`, { opacity: 1, ...enter }, '<0.1')
    .to(`.panel.${panel} > :not(.verdict)`, { opacity: 1, ...enter, stagger: 0.1 }, '<0.1')
    .fromTo(`.panel.${panel} > .verdict`, { y: -16 }, { opacity: 1, y: 0, ...pop }, '>0.2')
  relabel('judged', 'eager', 'eager', 'every')
  timeline.addLabel('step5')
  relabel('eager', 'careful', 'careful', 'eager')
  timeline.addLabel('step6')

  // Click 7. The three kinds of model sit side by side, and only the balanced one scores high.
  timeline
    .to('.panel.careful > *', { opacity: 0, ...exit })
    .to('.card.leaving', { opacity: 0, ...exit }, '<')
    .to('.card:not(.leaving)', { left: (index: number) => trioAt(index).left, top: (index: number) => trioAt(index).top, width: 130, height: 92, ...moveLong }, '<0.1')
    .to('.group-0 .layer.careful, .group-2 .layer.careful', { opacity: 0, ...exit }, '<0.2')
    .to('.group-0 .layer.eager, .group-2 .layer.balanced', { opacity: 1, ...enter }, '<0.1')
    .to('.trio-label', { opacity: 1, ...enter, stagger: 0.15 }, '>0.1')
    .to('.outline', { opacity: 1, ...enter }, '>')
    .to('.rule', { opacity: 1, ...enter }, '<0.1')
    .addLabel('step7')
})
</script>

<template>
  <div ref="root" class="f1">
    <div v-for="(block, index) in blocks" :key="`block-${index}`" class="block" :style="px(block)" />

    <div class="eyebrow one">One post</div>
    <div class="eyebrow model">Model's answers</div>
    <div class="eyebrow judged">Checked against the answer key</div>

    <div class="outline" :style="px({ left: trio[2].left - 12, top: -12, width: groupWidth + 24, height: 352 })" />

    <div
      v-for="(card, index) in cards" :key="`card-${index}`" class="card" :class="[card.group, { first: index === 0 }]"
      :style="px(cellAt(index))"
    >
      <div v-if="index === 0" class="big-text">
        <span v-for="(piece, part) in firstPost" :key="part" :class="piece.kind && ['k', piece.kind]">{{ piece.text }}<span v-if="piece.kind" class="sup">{{ sups[piece.kind] }}</span></span>
      </div>
      <div v-for="layer in card.layers" :key="layer.mode" class="layer" :class="layer.mode">
        <span v-for="(piece, part) in layer.pieces" :key="part" :class="piece.mark && ['mark', piece.mark]">{{ piece.text }}</span>
      </div>
    </div>

    <div class="legend">
      <span class="mark ok">Correct</span>
      <span class="mark no">Wrong</span>
      <span class="mark ms">Missed</span>
    </div>

    <div class="panel every">
      <div class="title">Every post is checked</div>
    </div>
    <div class="panel eager">
      <div class="title">An eager model</div>
      <div class="desc">Marks almost every word</div>
      <div class="what">Many wrong answers</div>
      <div class="score-name">F1-score</div>
      <div class="verdict">Low</div>
    </div>
    <div class="panel careful">
      <div class="title">A careful model</div>
      <div class="desc">Marks only what it is sure of</div>
      <div class="what">Many missed answers</div>
      <div class="score-name">F1-score</div>
      <div class="verdict">Low</div>
    </div>

    <div v-for="(label, group) in trio" :key="label.what" class="trio-label" :class="{ best: group === 2 }" :style="px({ left: label.left, width: groupWidth })">
      <div class="what">{{ label.what }}</div>
      <div class="score-name">F1-score</div>
      <div class="verdict">{{ label.verdict }}</div>
    </div>

    <p class="rule">High only when its answers are correct <span class="and">and</span> it finds the correct answers</p>
  </div>
</template>

<style scoped>
.f1 {
  --answer-fill: #fdf1e3;
  --answer-ink: #7c3a06;
  --correct-fill: #e9f6ee;
  --correct-line: #3f9a63;
  --correct-ink: #14532d;

  position: relative;
  height: 456px;
}

.block {
  position: absolute;
  width: 30px;
  height: 30px;
  border-radius: 4px;
  background: var(--color-data);
}

.eyebrow {
  position: absolute;
  left: 0;
  top: 14px;
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-subtle);
}

.card {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid var(--color-data);
  border-radius: 10px;
  background: var(--color-surface);
  overflow: hidden;
}

.first {
  overflow: visible;
}

.big-text {
  position: absolute;
  inset: 0;
  padding: 44px 48px;
  font-size: 2.5rem;
  line-height: 2.1;
  color: var(--color-ink);
}

.k {
  position: relative;
  display: inline-block;
  padding: 0 0.25em;
  line-height: 1.45;
  border: 0.1em solid transparent;
  border-radius: 0.3em;
  white-space: nowrap;
  text-decoration-line: line-through;
  text-decoration-color: transparent;
  text-decoration-thickness: 0.07em;
}

.k.m {
  border-style: dashed;
}

.sup {
  position: absolute;
  top: -0.75em;
  right: -0.4em;
  font-size: 0.45em;
  font-weight: 700;
  line-height: 1.2;
  padding: 0.1em 0.35em;
  border-radius: 0.6em;
  background: var(--color-surface);
}

.k.c .sup {
  color: var(--correct-line);
}

.k.w .sup,
.k.m .sup {
  color: var(--color-subtle);
}

.first .layer,
.layer {
  position: absolute;
  inset: 0;
  padding: 6px 7px;
  font-size: 12px;
  line-height: 1.65;
  color: var(--color-ink);
}

.mark {
  padding: 0.02em 0.25em;
  border: 0.13em solid transparent;
  border-radius: 0.45em;
  white-space: nowrap;
}

.mark.ok {
  background: var(--correct-fill);
  border-color: var(--correct-line);
  color: var(--correct-ink);
}

.mark.no {
  background: var(--color-land);
  border-color: var(--color-subtle);
  color: var(--color-muted);
  text-decoration: line-through;
  text-decoration-thickness: 0.08em;
}

.mark.ms {
  border: 0.13em dashed var(--color-subtle);
}

.legend {
  position: absolute;
  left: 0;
  right: 0;
  top: 384px;
  display: flex;
  justify-content: center;
  gap: var(--space-3);
  font-size: var(--text-xs);
  color: var(--color-muted);
}

.panel {
  position: absolute;
  left: 624px;
  top: 70px;
  width: 260px;
}

.panel.every {
  top: 190px;
}

.title {
  font-size: var(--text-xl);
  font-weight: 700;
  line-height: 1.2;
  color: var(--color-ink);
}

.desc {
  margin-top: var(--space-1);
  font-size: var(--text-sm);
  color: var(--color-muted);
}

.panel .what {
  margin-top: var(--space-3);
  padding-top: var(--space-2);
  border-top: 2px solid var(--color-subtle);
}

.what {
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--color-muted);
}

.score-name {
  margin-top: 6px;
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-subtle);
}

.verdict {
  font-size: var(--text-4xl);
  font-weight: 700;
  line-height: 1.1;
  color: var(--color-ink);
}

.trio-label {
  position: absolute;
  top: 214px;
  text-align: center;
}

.trio-label.best .what {
  color: var(--color-ink);
}

.outline {
  position: absolute;
  box-sizing: border-box;
  border: 3px solid var(--color-subtle);
  border-radius: 14px;
}

.rule {
  position: absolute;
  left: 0;
  right: 0;
  top: 384px;
  margin: 0;
  text-align: center;
  font-size: var(--text-md);
  font-weight: 600;
  color: var(--color-ink);
}

.and {
  color: var(--color-subtle);
}
</style>
