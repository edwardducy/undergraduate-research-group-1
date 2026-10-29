export const slide = { width: 1040, marginX: 78 }

export const model = { x: 442, top: 112, middle: 144, bottom: 176, width: 256 }
export const modelOrigin = `${model.x} ${model.middle}`

export const angle = 60
export const conflictAngle = 78
export const reach = 200
export const strongerReach = 300
export const shortReach = 130
export const easierAngle = (Math.acos(0.7) * 180) / Math.PI
export const harderAngle = (Math.acos(0.45) * 180) / Math.PI
export const sideX = { equal: 221, advanced: 663 }

export function tip(length: number, degrees = angle) {
  const radians = (degrees * Math.PI) / 180
  return { x: length * Math.sin(radians), y: length * Math.cos(radians) }
}

export const tilt = 5
const halfHeight = (model.bottom - model.top) / 2
export const pivot = {
  x: -halfHeight * Math.sin((tilt * Math.PI) / 180),
  y: halfHeight * Math.cos((tilt * Math.PI) / 180) - halfHeight,
}

export const approach = { modelWidth: model.width, examplesWidth: 393, modelTop: 144 }
const space = (slide.width - approach.modelWidth - approach.examplesWidth) / 3
export const approachModelX = -slide.marginX + space + approach.modelWidth / 2
export const approachExamplesX = -slide.marginX + 2 * space + approach.modelWidth
