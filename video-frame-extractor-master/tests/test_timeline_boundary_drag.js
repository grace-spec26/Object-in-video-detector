const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const htmlPath = path.join(__dirname, '..', 'index.html')
const html = fs.readFileSync(htmlPath, 'utf8')

for (const text of [
  '.timeline-boundary-handle',
  'function updateSegmentBoundary',
  'function startSegmentBoundaryDrag',
  'function dragSegmentBoundaryToClientX',
  'let draggingSegmentBoundary',
  "handle.className = 'timeline-boundary-handle'",
  'handle.dataset.boundaryIndex',
  "timelineBarContainer.addEventListener('pointermove'",
  "timelineBarContainer.addEventListener('pointerup'",
  "timelineBarContainer.addEventListener('pointercancel'",
]) {
  assert(html.includes(text), `Expected draggable segment boundary support: ${text}`)
}

assert(
  html.includes('cutPoints[boundaryIndex] = nextTime'),
  'Dragging in Cutting Points mode should update the internal cut point.',
)
assert(
  html.includes('explicitSegments[boundaryIndex - 1].end = nextTime') &&
    html.includes('explicitSegments[boundaryIndex].start = nextTime'),
  'Dragging in Explicit mode should update the adjacent segment end/start together.',
)
