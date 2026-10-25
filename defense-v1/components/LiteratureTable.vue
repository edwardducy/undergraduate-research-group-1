<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The same motion tokens as slide 2
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }

// Studies not being discussed fade back but stay readable
const dim = 0.35

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const land = getComputedStyle(document.documentElement).getPropertyValue('--color-land').trim()

  // Each row reads as a chain, what the study did, what it left open, and how this study answers it
  gsap.set('tbody .points, tbody .ours', { opacity: 0 })

  const studies = ['ali', 'caoile', 'kurin']
  let step = 0
  timeline.addLabel('step0')

  // Each study takes three clicks, and only the cell being discussed is at full strength
  studies.forEach((study, index) => {
    const others = studies
      .filter(other => other !== study)
      .map(other => studies.indexOf(other) < index ? `.${other} td` : `.${other} .study, .${other} .did`)
      .join(', ')

    timeline
      .to(others, { opacity: dim, ...exit })
      .to(`.${study} .study, .${study} .did`, { opacity: 1, ...enter }, '<')
      .addLabel(`step${++step}`)
      .to(`.${study} .did`, { opacity: dim, ...exit })
      .to(`.${study} .points`, { opacity: 1, ...enter }, '<0.15')
      .addLabel(`step${++step}`)
      .to(`.${study} .points`, { opacity: dim, ...exit })
      .to(`.${study} .ours`, { opacity: 1, ...enter }, '<0.15')
      .addLabel(`step${++step}`)
  })

  // Together these form the gap, so this study's column comes into focus
  timeline
    .to('td.ours', { opacity: 1, backgroundColor: land, ...enter })
    .to('th.ours', { backgroundColor: land, ...enter }, '<')
    .to('td:not(.ours)', { opacity: dim, ...exit }, '<')
    .addLabel(`step${++step}`)
})
</script>

<template>
  <div ref="root" class="literature">
    <table>
      <colgroup>
        <col class="study-column">
        <col>
        <col>
        <col>
      </colgroup>
      <thead>
        <tr>
          <th>Study</th>
          <th>What it did</th>
          <th class="points">Left open</th>
          <th class="ours">This study</th>
        </tr>
      </thead>
      <tbody>
        <tr class="ali">
          <td class="study">Ali et al. (2026)</td>
          <td class="did">One shared encoder for both tasks, on English disaster posts</td>
          <td class="points">Other languages, and PCGrad combined with uncertainty weighting</td>
          <td class="ours">Taglish posts, with the combined optimizer among the 13</td>
        </tr>
        <tr class="caoile">
          <td class="study">Caoile and Aviles (2025)</td>
          <td class="did">Large language model, on Filipino, Ilocano, and Taglish comments</td>
          <td class="points">Whether a smaller trained model can give both outputs</td>
          <td class="ours">A smaller model trained on labeled Taglish posts</td>
        </tr>
        <tr class="kurin">
          <td class="study">Kurin et al. (2022)</td>
          <td class="did">Equal weighting compared with advanced optimizers, on images and robot tasks</td>
          <td class="points">Settings where equal weighting may do worse</td>
          <td class="ours">Taglish text with a <span class="whole">whole-post task</span> and a <span class="word">per-word task</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.literature {
  height: 100%;
}

table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
}

.study-column {
  width: 16%;
}

th,
td {
  padding: var(--space-1) var(--space-2) var(--space-1) var(--space-1);
  text-align: left;
  vertical-align: top;
}

/* Headers share the uppercase label style of the diagram's tags */
th {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  border-bottom: 2px solid var(--color-ink);
}

td {
  font-size: var(--text-2xs);
  line-height: var(--leading-list);
  color: var(--color-ink);
  border-bottom: 1px solid var(--color-land);
  padding-top: var(--space-2);
  padding-bottom: var(--space-2);
}

.study {
  font-weight: 600;
}

.whole,
.word {
  font-weight: 600;
}

.whole {
  color: var(--color-classification);
}

.word {
  color: var(--color-ner);
}
</style>
