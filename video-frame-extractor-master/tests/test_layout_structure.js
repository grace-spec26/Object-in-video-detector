const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const htmlPath = path.join(__dirname, '..', 'index.html')
const html = fs.readFileSync(htmlPath, 'utf8')

function indexOfOrFail(source, text) {
  const index = source.indexOf(text)
  assert.notEqual(index, -1, `Expected to find ${text}`)
  return index
}

const workspaceStart = indexOfOrFail(html, '<div class="video-workspace hidden" id="videoWorkspace">')
const workspaceEnd = indexOfOrFail(html.slice(workspaceStart), '<div class="snapshot-section hidden" id="snapshotSection">') + workspaceStart
const workspace = html.slice(workspaceStart, workspaceEnd)

assert(
  workspace.indexOf('<div class="video-main-column" id="videoMainColumn">') <
    workspace.indexOf('<aside class="segments-section" id="segmentsPanel">'),
  'Video/detector column should appear before the right-side segments panel',
)

const videoColumn = workspace.slice(
  indexOfOrFail(workspace, '<div class="video-main-column" id="videoMainColumn">'),
  indexOfOrFail(workspace, '<aside class="segments-section" id="segmentsPanel">'),
)

for (const id of ['videoPreview', 'sceneChartSection', 'timelineSection']) {
  assert(videoColumn.includes(`id="${id}"`), `Expected ${id} inside videoMainColumn`)
}

const segmentsPanel = workspace.slice(indexOfOrFail(workspace, '<aside class="segments-section" id="segmentsPanel">'))
for (const id of ['segmentList', 'addSegmentBottomBtn']) {
  assert(segmentsPanel.includes(`id="${id}"`), `Expected ${id} inside segmentsPanel`)
}

assert(html.includes('.video-workspace {'), 'Expected video-workspace CSS')
assert(html.includes('padding: 10px;'), 'Expected a compact outer purple boundary')
assert(html.includes('max-width: 1800px;'), 'Expected a wide desktop workspace')
assert(html.includes('grid-template-columns: minmax(0, 1fr) minmax(300px, 360px)'), 'Expected a space-efficient desktop workspace grid')
assert(html.includes('max-height: 72vh;'), 'Expected the video to use available viewport space')
assert(html.includes('.video-workspace.hidden {'), 'Expected hidden workspace override')
assert(html.includes('@media (max-width: 900px)'), 'Expected responsive single-column workspace media query')
assert(html.includes('@media (max-width: 600px)'), 'Expected compact segment controls on narrow screens')
assert(html.includes("const videoWorkspace = document.getElementById('videoWorkspace')"), 'Expected videoWorkspace DOM reference')
assert(html.includes("videoWorkspace.classList.remove('hidden')"), 'Expected workspace to be shown when video loads')
