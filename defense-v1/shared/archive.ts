// The X API box on the X API slide, which also shows the pilot search.
export const archive = { x: 540, y: 60, width: 300, height: 290 }
export const post = { width: 56, height: 40, gap: 12 }

export const cells = Array.from({ length: 12 }, (_, index) => ({
  x: archive.x + 20 + (index % 4) * (post.width + post.gap),
  y: archive.y + 100 + Math.floor(index / 4) * (post.height + post.gap),
}))

export const lock = { x: archive.x + archive.width - 38, y: archive.y + 30, open: -28 }
// The shackle turns about its left foot.
export const hinge = { x: lock.x - 9, y: lock.y + 2 }
