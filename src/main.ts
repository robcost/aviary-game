/**
 * Aviary — a first-person Australian bird flight simulator.
 *
 * Entry point. Builds the WebGL renderer, the shared {@link GameContext},
 * and the ordered module list, then drives the frame loop.
 *
 * URL parameters (used by the review harness and for deep links):
 * - `?bird=<SpeciesId>` — preselect a species.
 * - `?autofly=1` — skip the menu and fly a scripted showcase path.
 * - `?shot=N` — photo mode: freeze at preset camera N (1-based).
 * - `?debug=1` — show a performance overlay (draw calls, triangles, frame time).
 */
import * as THREE from 'three'
import { WORLD, type GameContext, type GameModule, type SpeciesId, SPECIES_INFO } from './core/GameState'
import { Sky } from './world/Sky'
import { Terrain } from './world/Terrain'
import { Water } from './world/Water'
import { Vegetation } from './world/Vegetation'
import { Birds } from './birds/Birds'
import { Flight } from './flight/Flight'
import { AudioSystem } from './audio/AudioSystem'
import { PostProcessing } from './fx/PostProcessing'
import { UI } from './ui/UI'

const params = new URLSearchParams(location.search)

/** Parse a `?bird=` value, falling back to the eagle. */
function parseSpecies(value: string | null): SpeciesId {
  return value && value in SPECIES_INFO ? (value as SpeciesId) : 'wedge-tailed-eagle'
}

/**
 * Performance overlay for `?debug=1`. Prints whole-frame draw calls,
 * triangle count, GPU resource counts, fps, and CPU frame time so art
 * reviews can verify the render budget.
 *
 * The renderer's `info` counters normally reset on every `render()` call,
 * which under-reports a multi-pass composer frame. The overlay disables
 * `autoReset` and resets the counters once per frame instead, so the
 * numbers cover every pass. Text updates are throttled to 2 Hz; between
 * updates the frame loop only accumulates plain numbers.
 */
class DebugOverlay {
  private readonly el: HTMLDivElement
  private frames = 0
  private cpuMsSum = 0
  private cpuMsWorst = 0
  private elapsed = 0

  constructor(target: THREE.WebGLRenderer) {
    target.info.autoReset = false
    this.el = document.createElement('div')
    this.el.style.cssText =
      'position:fixed;top:8px;left:8px;z-index:1000;padding:6px 10px;' +
      'font:12px/1.5 ui-monospace,Menlo,monospace;color:#8f8;' +
      'background:rgba(0,0,0,0.65);border-radius:4px;pointer-events:none;white-space:pre'
    document.body.appendChild(this.el)
  }

  /** Call at the top of the frame, before any module renders. */
  beginFrame(target: THREE.WebGLRenderer): void {
    target.info.reset()
  }

  /** Call after the final render pass with the measured CPU frame time. */
  endFrame(dt: number, cpuMs: number, target: THREE.WebGLRenderer): void {
    this.frames++
    this.cpuMsSum += cpuMs
    if (cpuMs > this.cpuMsWorst) this.cpuMsWorst = cpuMs
    this.elapsed += dt
    if (this.elapsed < 0.5) return
    const r = target.info.render
    const m = target.info.memory
    this.el.textContent =
      `fps        ${(this.frames / this.elapsed).toFixed(1)}\n` +
      `cpu ms     ${(this.cpuMsSum / this.frames).toFixed(2)} avg  ${this.cpuMsWorst.toFixed(2)} max\n` +
      `draw calls ${r.calls}\n` +
      `triangles  ${r.triangles.toLocaleString('en')}\n` +
      `geometries ${m.geometries}   textures ${m.textures}\n` +
      `programs   ${target.info.programs?.length ?? 0}`
    this.frames = 0
    this.cpuMsSum = 0
    this.cpuMsWorst = 0
    this.elapsed = 0
  }
}

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
renderer.setSize(innerWidth, innerHeight)
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.0
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
// The canvas stays hidden until the first full frame has rendered. An
// unrendered WebGL canvas is opaque black; without this it would cover the
// dawn boot backdrop (the body's --boot-sky) from script eval until the
// UI module builds its boot-fade overlay, putting a black screen back into
// the load sequence. Rendering into a hidden canvas works normally, so the
// reveal below is free.
renderer.domElement.style.visibility = 'hidden'
document.getElementById('app')!.appendChild(renderer.domElement)

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(75, innerWidth / innerHeight, 0.05, 40000)
scene.add(camera)

const ctx: GameContext = {
  scene,
  camera,
  renderer,
  time: 0,
  species: parseSpecies(params.get('bird')),
  flying: false,
  flight: {
    position: new THREE.Vector3(WORLD.spawn.x, WORLD.spawn.y, WORLD.spawn.z),
    velocity: new THREE.Vector3(0, 0, -14),
    quaternion: new THREE.Quaternion(),
    speed: 14,
    altitude: WORLD.spawn.y,
    flapPhase: 0,
    flapStrength: 0,
    banking: 0,
    stalled: false,
  },
  sunDirection: new THREE.Vector3(0.35, 0.4, -0.55).normalize(),
  getTerrainHeight: () => 0,
  photoShot: params.get('shot') ? Number(params.get('shot')) : null,
  autofly: params.get('autofly') === '1',
  paused: false,
  events: new EventTarget(),
}

if (ctx.autofly || ctx.photoShot !== null) ctx.flying = true

ctx.events.addEventListener('start-flight', (e) => {
  ctx.species = (e as CustomEvent<{ species: SpeciesId }>).detail.species
  ctx.flying = true
})

const fx = new PostProcessing()
const modules: GameModule[] = [
  new Flight(),
  new Sky(),
  new Terrain(),
  new Water(),
  new Vegetation(),
  new Birds(),
  new AudioSystem(),
  new UI(),
  fx,
]

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
})

/** Init every module in order, then start the frame loop. */
async function boot(): Promise<void> {
  const logBoot = params.get('debug') === '1'
  const t0 = performance.now()
  let warming = false
  for (const m of modules) {
    const t = performance.now()
    try {
      await m.init(ctx)
    } catch (err) {
      // A module init failure is fatal: later modules never init and the
      // frame loop never starts. Name the module loudly instead of letting
      // an anonymous unhandled rejection abort the pipeline in silence.
      console.error(`[boot] FATAL: module "${m.name}" failed to init — scene boot aborted`, err)
      throw err
    }
    if (logBoot) console.debug(`[boot] init ${m.name}: ${(performance.now() - t).toFixed(0)}ms`)
    // Pipelined shader warm-up. The Sky module builds the entire light rig
    // (sun + hemisphere), so from its init onward the light configuration —
    // and therefore every program's cache key — is final. Kick a
    // non-awaited compileAsync after each module from that point: the
    // parallel-compile extension moves the heavy sky/terrain/water shader
    // compiles onto driver threads WHILE later modules run their CPU-side
    // init, instead of serializing compile after the whole init chain. On
    // slow hardware this cuts whole seconds off the boot hold. Errors are
    // swallowed here; the awaited compile below reports them.
    if (warming || m.name === 'sky') {
      warming = true
      renderer.compileAsync(scene, camera).catch(() => {})
    }
  }

  // Barrier before the first visible frame: re-traverse and wait until
  // every scene shader (including any added after the last warm-up kick)
  // has finished compiling, instead of stalling frame 1 for seconds.
  // Already-compiled programs resolve from the cache immediately. The
  // canvas is still hidden and the boot backdrop is on screen, so any
  // remaining upload hitch happens behind the styled hold, never as a
  // black or half-drawn frame.
  await renderer.compileAsync(scene, camera)
  if (logBoot) console.debug(`[boot] total boot: ${(performance.now() - t0).toFixed(0)}ms`)

  const overlay = params.get('debug') === '1' ? new DebugOverlay(renderer) : null
  const clock = new THREE.Clock()
  let firstFrameDone = false
  renderer.setAnimationLoop(() => {
    overlay?.beginFrame(renderer)
    const frameStart = overlay ? performance.now() : 0
    const dt = Math.min(clock.getDelta(), 0.05)
    if (!ctx.paused) ctx.time += dt
    for (const m of modules) m.update(dt, ctx)
    fx.render(dt)
    if (!firstFrameDone) {
      // The full pipeline (composer passes included) has now rendered once.
      // Reveal the canvas (hidden since creation — see above) and tell the
      // UI, which starts its capped 1.5 s dissolve from the boot backdrop.
      // The reveal always lands on a finished frame; in photo mode (no boot
      // fade) the canvas simply appears fully rendered.
      firstFrameDone = true
      renderer.domElement.style.visibility = 'visible'
      ctx.events.dispatchEvent(new Event('first-frame'))
    }
    overlay?.endFrame(dt, performance.now() - frameStart, renderer)
  })
}

boot().catch((err: unknown) => {
  // Hard failure path: the error (already logged with its module name above,
  // or raised by the compile barrier) leaves the scene incomplete. Surface
  // it once more at top level so a review console can never miss it.
  console.error('[boot] FATAL: Aviary failed to boot.', err)
})

declare global {
  interface Window {
    /** Debug handle for the review harness. */
    __aviary: GameContext
  }
}
window.__aviary = ctx
