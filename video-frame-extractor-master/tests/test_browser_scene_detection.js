const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

const htmlPath = path.join(__dirname, '..', 'index.html')
const html = fs.readFileSync(htmlPath, 'utf8')
const match = html.match(/function percentile[\s\S]*?(?=\n\s+function mergeCameraSwitchCutPoints)/)

assert(match, 'Could not find browser scene detection functions in index.html')

const context = {}
vm.createContext(context)
vm.runInContext(match[0], context)

function samplesFromDeltas(deltas, interval = 1) {
  return deltas.map((delta, index) => ({ time: index * interval, delta }))
}

{
  const deltas = [
    0,
    12, 13, 12, 14,
    32,
    14, 13, 12,
    45, 47, 46, 48, 47, 46, 49, 47, 48, 46, 47, 48, 46, 47,
    13, 12,
    34,
    13, 12,
    0,
  ]

  const switches = context.detectCameraSwitchTimes(samplesFromDeltas(deltas), 29, 1)

  assert(
    switches.some(time => Math.abs(time - 5) < 0.001),
    `expected isolated local peak at 5s to be detected, got ${JSON.stringify(switches)}`,
  )
  assert(
    switches.some(time => Math.abs(time - 25) < 0.001),
    `expected isolated local peak at 25s to be detected, got ${JSON.stringify(switches)}`,
  )
}

{
  const deltas = [
    0,
    11, 12, 11, 12, 13, 12, 11,
    12, 13, 12, 11, 12, 13, 12,
    0,
  ]

  const switches = context.detectCameraSwitchTimes(samplesFromDeltas(deltas), 16, 1)

  assert.equal(switches.length, 0, `expected smooth motion not to create switches, got ${JSON.stringify(switches)}`)
}
