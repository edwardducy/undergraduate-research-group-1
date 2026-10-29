import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { onMounted, onUnmounted, watch, type Ref } from 'vue'

export const stepCount = (timeline: gsap.core.Timeline) => Object.keys(timeline.labels).filter(label => label.startsWith('step')).length
export const addStep = (timeline: gsap.core.Timeline) => timeline.addLabel(`step${stepCount(timeline)}`)

export function useClickTimeline(scope: Ref<Element | undefined>, build: (timeline: gsap.core.Timeline) => void, first = 0, rewind = false) {
  const { $clicks } = useSlideContext()

  let context: gsap.Context
  let timeline: gsap.core.Timeline
  let move: gsap.core.Tween | undefined
  let last = 0

  const stepAt = (clicks: number) => Math.min(Math.max(clicks - first, 0), last)

  onMounted(() => {
    context = gsap.context(() => {
      timeline = gsap.timeline({ paused: true, defaults: { duration: 0.5, ease: 'power2.out', autoRound: false } })
      build(timeline)
    }, scope.value)

    last = stepCount(timeline) - 1
    timeline.seek(`step${stepAt($clicks.value)}`)
  })

  watch($clicks, (clicks, previousClicks) => {
    const step = stepAt(clicks)
    const previous = stepAt(previousClicks)
    if (step === previous)
      return
    move?.kill()
    if (step === previous + 1 || (rewind && step === previous - 1)) {
      timeline.seek(`step${previous}`)
      move = timeline.tweenTo(`step${step}`)
      if (step < previous)
        move.timeScale(2)
    }
    else
      timeline.seek(`step${step}`)
  })

  onUnmounted(() => context.revert())
}
