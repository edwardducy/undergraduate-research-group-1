<script setup lang="ts">
import { onSlideEnter, useIsSlideActive } from '@slidev/client'
import { type GeoJSONSource, Map, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import worker from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import data from '../data/marikina-posts.json'

const SLOTS: Record<string, string> = {
  'Requests or urgent needs': 'series-1',
  'Infrastructure and utility damage': 'series-2',
  'Displaced people and evacuations': 'series-3',
  'Rescue, volunteering, or donation effort': 'series-4',
  'Sympathy and support': 'series-5',
  'Missing or found people': 'series-6',
  'Caution and advice': 'series-7',
  'Injured or dead people': 'series-8',
  'Other relevant information': 'series-9',
  'Not humanitarian': 'series-10',
}
const slot = (category: string) => SLOTS[category]
const swatch = (category: string) => `var(--app-${slot(category)})`

const CYCLE = 20
const ARRIVE = 0.5
const ARRIVED = 15
const FADE = 1
const OPENS = [3.5, 6.5, 9.5, 12.5, 15.2]
const LEAD = 0.4

type Post = { lat: number | null, lon: number | null, category: number, at: number, text?: string }

const posts: Post[] = [
  ...data.stream.map(([lat, lon, category], index) => ({
    lat,
    lon,
    category: category as number,
    at: ARRIVE + (ARRIVED - ARRIVE) * ((index + 1) / data.stream.length) ** (1 / 3),
  })),
  ...data.featured.map((post, index) => ({ ...post, at: OPENS[index] - LEAD })),
].sort((a, b) => a.at - b.at)
const featured = posts.filter(post => post.text)

const finals = data.categories.map((_, index) => posts.filter(post => post.category === index).length)
const order = data.categories.map((category, index) => ({ category, index })).sort((a, b) => finals[b.index] - finals[a.index])
const most = Math.max(...finals)

const shown = ref(data.categories.map(() => 0))
const located = ref(0)
const unlocated = ref(0)
const opened = ref<Post>()
const openedCategory = computed(() => data.categories[opened.value?.category ?? 0])

const view: [[number, number], [number, number]] = [[121.074, 14.612], [121.136, 14.684]]
const saved: [[number, number], [number, number]] = [[data.bounds[0][1], data.bounds[0][0]], [data.bounds[1][1], data.bounds[1][0]]]

setWorkerUrl(worker)

const root = ref<HTMLElement>()
const container = ref<HTMLElement>()
const card = ref<HTMLElement>()
let map: Map | undefined
let popup: Popup | undefined
let observer: ResizeObserver | undefined
let frame = 0

const active = useIsSlideActive()
let started = performance.now()
onSlideEnter(() => {
  started = performance.now()
})
const still = matchMedia('(prefers-reduced-motion: reduce)').matches

const list = ref<HTMLElement>()
const above = ref(false)
const below = ref(false)
function edges() {
  const { scrollTop, scrollHeight, clientHeight } = list.value!
  above.value = scrollTop > 1
  below.value = scrollTop + clientHeight < scrollHeight - 1
}

onMounted(async () => {
  const base = new URL(import.meta.env.BASE_URL, location.href).href
  const style = JSON.parse((await (await fetch(`${base}map/style.json`)).text()).replaceAll('{base}', base))

  const css = getComputedStyle(root.value!)
  const token = (name: string) => css.getPropertyValue(`--app-${name}`).trim()
  const colour = ['match', ['get', 'category'], ...data.categories.flatMap((category, index) => [index, token(slot(category))]), token('series-10')]

  map = new Map({
    container: container.value!,
    style,
    bounds: view,
    maxBounds: saved,
    maxZoom: 16,
    keyboard: false,
    dragRotate: false,
    pitchWithRotate: false,
    touchPitch: false,
    attributionControl: { compact: false },
  })
  map.touchZoomRotate.disableRotation()
  map.addControl(new NavigationControl({ showCompass: false }), 'top-right')

  map.on('load', () => {
    map!.addSource('posts', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: posts.filter(post => post.lat !== null).map(post => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [post.lon!, post.lat!] },
          properties: { category: post.category, at: post.at },
        })),
      },
    })
    map!.addLayer({
      id: 'posts',
      type: 'circle',
      source: 'posts',
      filter: ['<=', ['get', 'at'], 0],
      paint: {
        'circle-radius': 4,
        'circle-color': colour as never,
        'circle-stroke-color': token('surface'),
        'circle-stroke-width': 1,
      },
    })

    map!.addSource('opened', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
    map!.addLayer({ id: 'opened-halo', type: 'circle', source: 'opened', paint: { 'circle-radius': 7, 'circle-opacity': 0 } })
    map!.addLayer({
      id: 'opened',
      type: 'circle',
      source: 'opened',
      paint: { 'circle-radius': 7, 'circle-stroke-color': token('surface'), 'circle-stroke-width': 2 },
    })
    popup = new Popup({ closeButton: false, closeOnClick: false, anchor: 'top', offset: 16, maxWidth: '250px' })
      .setDOMContent(card.value!)

    const show = (post: Post | undefined) => {
      opened.value = post
      const source = map!.getSource('opened') as GeoJSONSource
      if (!post) {
        source.setData({ type: 'FeatureCollection', features: [] })
        popup!.remove()
        return
      }
      const own = token(slot(data.categories[post.category]))
      map!.setPaintProperty('opened-halo', 'circle-color', own)
      map!.setPaintProperty('opened', 'circle-color', own)
      source.setData({ type: 'Point', coordinates: [post.lon!, post.lat!] })
      popup!.setLngLat([post.lon!, post.lat!]).addTo(map!)
    }

    let last: Post | undefined
    let lastTime = -1
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick)
      if (!active.value)
        return
      const t = still ? CYCLE - FADE - 0.01 : ((now - started) / 1000) % CYCLE
      const fade = Math.min(1, Math.max(0, (CYCLE - t) / FADE))

      if (t !== lastTime) {
        map!.setFilter('posts', ['<=', ['get', 'at'], t])
        map!.setPaintProperty('posts', 'circle-radius', ['interpolate', ['linear'], ['get', 'at'], t - 0.35, 4, t - 0.15, 5.5, t, 0])
        map!.setPaintProperty('posts', 'circle-opacity', fade)
        map!.setPaintProperty('posts', 'circle-stroke-opacity', fade)
        lastTime = t
      }
      const counts = data.categories.map(() => 0)
      let onMap = 0
      for (const post of posts) {
        if (post.at > t)
          break
        counts[post.category]++
        if (post.lat !== null)
          onMap++
      }
      shown.value = counts.map(count => Math.round(count * fade))
      located.value = Math.round(onMap * fade)
      unlocated.value = Math.round((counts.reduce((sum, count) => sum + count, 0) - onMap) * fade)

      const index = OPENS.findLastIndex(time => time <= t)
      const post = index >= 0 ? featured[index] : undefined
      if (post !== last) {
        show(post)
        last = post
      }
      if (post) {
        const since = Math.min(1, (t - OPENS[index]) / 0.25)
        map!.setPaintProperty('opened', 'circle-radius', 4 + 3 * (1 - (1 - since) ** 3))
        map!.setPaintProperty('opened', 'circle-opacity', fade)
        map!.setPaintProperty('opened', 'circle-stroke-opacity', fade)
        popup!.getElement().style.opacity = String(fade)
        const phase = still ? 0.5 : (now % 2400) / 2400
        map!.setPaintProperty('opened-halo', 'circle-radius', 7 + 13 * (1 - (1 - phase) ** 3))
        map!.setPaintProperty('opened-halo', 'circle-opacity', 0.4 * (1 - phase) * fade)
      }
    }
    frame = requestAnimationFrame(tick)
  })

  observer = new ResizeObserver(() => {
    map?.resize()
    edges()
  })
  observer.observe(container.value!)
  observer.observe(list.value!)
})

onUnmounted(() => {
  cancelAnimationFrame(frame)
  observer?.disconnect()
  map?.remove()
})
</script>

<template>
  <div ref="root" class="app">
    <section ref="container" class="map" />

    <div class="hidden">
      <div ref="card" class="card">
        <p>{{ opened?.text }}</p>
        <span class="category"><i :style="{ background: swatch(openedCategory) }" />{{ openedCategory }}</span>
      </div>
    </div>

    <aside>
      <h4>Posts per category</h4>
      <div ref="list" class="list" :class="{ above, below }" @scroll="edges">
        <div v-for="row in order" :key="row.category" class="bar-row">
          <span class="label"><i :style="{ background: swatch(row.category) }" />{{ row.category }}</span>
          <span class="count">{{ shown[row.index] }}</span>
          <span class="bar"><span :style="{ width: `${(shown[row.index] / most) * 100}%`, background: swatch(row.category) }" /></span>
        </div>
      </div>

      <div class="placed">
        <div><span>On the map</span><b>{{ located }}</b></div>
        <div><span>Without a location</span><b>{{ unlocated }}</b></div>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.app {
  --app-font: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --app-text-sm: 12px;
  --app-text-md: 13px;
  --app-radius: 6px;
  --app-surface: #ffffff;
  --app-panel: #f6f7f9;
  --app-border: #d8dce1;
  --app-text: #1f2328;
  --app-text-muted: #5c6670;
  --app-track: #e6e9ed;
  --app-series-1: #2a78d6;
  --app-series-2: #eb6834;
  --app-series-3: #1baf7a;
  --app-series-4: #eda100;
  --app-series-5: #e87ba4;
  --app-series-6: #008300;
  --app-series-7: #6250d6;
  --app-series-8: #a8233c;
  --app-series-9: #7a5230;
  --app-series-10: #9aa1a9;

  display: grid;
  grid-template-columns: 1fr 250px;
  height: 456px;
  border: 1px solid var(--app-border);
  border-radius: 8px;
  overflow: hidden;
  font-family: var(--app-font);
  font-size: var(--app-text-md);
  line-height: 1.4;
  color: var(--app-text);
  background: var(--app-surface);
}

.map {
  min-width: 0;
}

.hidden {
  display: none;
}

.card {
  display: grid;
  gap: 8px;
  font-family: var(--app-font);
  font-size: var(--app-text-md);
  line-height: 1.4;
  color: var(--app-text);
}

p {
  margin: 0;
  font-size: inherit;
  line-height: inherit;
}

.list {
  --fade: 28px;
  display: grid;
  align-content: start;
  gap: 14px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: none;
  overscroll-behavior: contain;
}

.list::-webkit-scrollbar {
  display: none;
}

.list.below {
  mask-image: linear-gradient(to bottom, black calc(100% - var(--fade)), transparent);
}

.list.above {
  mask-image: linear-gradient(to bottom, transparent, black var(--fade));
}

.list.above.below {
  mask-image: linear-gradient(to bottom, transparent, black var(--fade), black calc(100% - var(--fade)), transparent);
}

.bar-row {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 4px 8px;
  font-size: var(--app-text-sm);
}

.category,
.label {
  display: flex;
  align-items: center;
  gap: 6px;
}

i {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  transform: translateY(-0.075em);
}

.category {
  font-size: var(--app-text-sm);
  font-weight: 600;
}

aside {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
  padding: 14px 16px;
  border-left: 1px solid var(--app-border);
  background: var(--app-panel);
}

h4 {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.bar {
  grid-column: 1 / -1;
  height: 6px;
  border-radius: 4px;
  background: var(--app-track);
}

.bar span {
  display: block;
  height: 100%;
  border-radius: 4px;
}

.count {
  color: var(--app-text-muted);
  font-variant-numeric: tabular-nums;
}

.placed {
  display: grid;
  gap: 4px;
  font-size: var(--app-text-sm);
}

.placed div {
  display: flex;
  justify-content: space-between;
}

.placed span {
  color: var(--app-text-muted);
}

.map :deep(.maplibregl-ctrl-attrib) {
  font-family: var(--app-font);
  font-size: 10px;
}

.map :deep(.maplibregl-popup-content) {
  padding: 10px 12px;
  border-radius: var(--app-radius);
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
}
</style>
