export const benchmarkX = 150
export const benchmarkReach = 150
export const benchmarkChart = { left: 350, width: 510, top: 104, base: 368 }

const negativeTransfer = [0.15, 0.7, 0.35, 0.55, 0.1, 0.8, 0.25, 0.6, 0.3, 0.9, 0.2, 0.65, 0.4]
const cost = [0.15, 0.35, 0.5, 0.3, 0.85, 0.4, 0.95, 0.25, 0.6, 0.45, 0.9, 0.55, 0.7]
const height = benchmarkChart.base - benchmarkChart.top

export const benchmarkColumns = negativeTransfer.map((loss, index) => ({
  x: benchmarkChart.left + (benchmarkChart.width / negativeTransfer.length) * (index + 0.5),
  loss: loss * height,
  cost: cost[index] * height,
}))
