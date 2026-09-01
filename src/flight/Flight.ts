/**
 * Flight — the aerodynamic heart of Aviary.
 *
 * Owns {@link GameContext.flight} and the first-person camera. Implements a
 * simplified but convincing bird flight model:
 *
 * - Lift ∝ airspeed² × angle of attack; induced + parasitic drag; gravity;
 *   flap thrust delivered as beat-synced downstroke impulses (hold Space).
 * - Energy management: dive to trade altitude for speed, zoom-climb it back.
 * - Stall below minimum airspeed: buffet shake, a forced nose drop, and a
 *   natural speed-gaining recovery.
 * - Coordinated banked turns: roll into the turn and the yaw follows from the
 *   tilted lift vector, with a small tail-assist so slow turns stay honest.
 * - Environmental air: thermal columns over the arid eastern plateau, ridge
 *   lift on the windward slopes of the north-west range, simplex gusts on a
 *   prevailing onshore sea breeze, and a ground-effect cushion.
 * - Soft terrain/water collision: kill the into-surface velocity, slide with
 *   friction, thud the camera. Graze, never teleport.
 *
 * Controls: click the canvas for pointer-lock mouse (x = bank, y = pitch),
 * WASD/arrows fallback, Space = flap, Shift = tuck dive.
 *
 * Also drives three non-interactive camera modes:
 * - a slow cinematic drift while the menu is up,
 * - the `?autofly=1` showcase: a variable-speed sweep down the valley, low
 *   along the surf, up through plateau thermal country, and over the range,
 * - the four `?shot=N` photo presets (physics frozen; world keeps animating).
 *
 * The per-frame path allocates nothing: every vector is a module scratch.
 */
import * as THREE from 'three'
import { createNoise2D, type NoiseFunction2D } from 'simplex-noise'
import { WORLD, type GameContext, type GameModule, type SpeciesId } from '../core/GameState'

/** Gravitational acceleration, m/s². */
const G = 9.81

/** Altitude (m ASL) where thin air begins to erode lift, forming a soft ceiling. */
const THIN_AIR_START = 2400

/** Prevailing sea breeze, m/s: onshore from the southern ocean, slightly westerly. */
const WIND_BASE_X = -1.1
const WIND_BASE_Z = -4.2

/** Half-extent (m) inside which flight is unconstrained; beyond it we steer back. */
const SOFT_BOUND = WORLD.size / 2 - 500

/** Field-of-view range, degrees (vertical): a stroll → flat-out dive. Kept
 * deliberately tight — the first-person rig is parented to the camera, and a
 * wide FOV smears that near-field geometry into a distorted screen border.
 * Round-2 review still read the rig as a decorated border at 60–75°, so the
 * band drops again: 56° vertical ≈ 87° horizontal at 16:9 in cruise, and the
 * dive end now stays at 70° (~101° horizontal), safely under the fisheye
 * threshold. Together with the rig anchor sitting low/back (Birds module),
 * this keeps the wings in roughly the bottom third of the frame. */
const FOV_MIN = 56
const FOV_MAX = 70

/**
 * Soaring wing-beat frequency, Hz. Even a committed glide carries a slow
 * micro-oscillation at this rate — real wings trim continuously and are never
 * parked. Full-power flapping blends up to the species' own {@link SpeciesTuning.flapHz}.
 */
const SOAR_HZ = 1.35

/**
 * Visual wing-drive floor in a glide, before turbulence flex is added.
 * The rig scales its shoulder/elbow/wrist stroke by this value (the shoulder
 * swings ≈ ±0.95 rad × strength), so the floor must be high enough that the
 * {@link SOAR_HZ} trim cycle moves the shoulder through ~15° even in the
 * calmest glide — a parked-looking planar spread means this floor was too
 * timid. 0.26 gives ≈ ±14° at the shoulder before any turbulence is added.
 */
const WING_LIFE_MIN = 0.26

/**
 * Ceiling for the *visual* glide drive so an un-powered burst still reads
 * clearly below a committed full-power beat (which is exactly 1).
 */
const WING_LIFE_MAX = 0.85

/**
 * Gain from the turbulence rock signal into {@link FlightState.banking}.
 * The rig converts banking into asymmetric dihedral at roughly 0.2–1.0 rad
 * per unit across the elbow/wrist chain, so the wander must regularly reach
 * ±0.3 or more for the left/right wings to differ visibly in a still frame.
 * The round-1 value (0.22 on a ±0.85 signal → ±0.19 → ~2° at the wrist)
 * was invisible in captures. Round 2: the rock is now the multi-band
 * {@link Flight.rockSignal} (peak ≈ ±2.0, typical ±0.9), and this gain puts
 * ±0.3..0.55 of asymmetric drive on even a dead-straight cruise, with a
 * guaranteed 1.3 Hz component so no two frames seconds apart ever match.
 * Note {@link FlightState.banking} does NOT roll the camera — horizon roll
 * comes from the physical bank in composePose — so a generous rock here
 * animates the wings without any drunk-horizon side effect.
 */
const ROCK_GAIN = 0.4

/**
 * Per-species handling model. Every aerodynamic coefficient is derived from
 * these physical numbers in {@link Flight.applySpecies}, so the tuning table
 * reads in real units and stays honest.
 */
interface SpeciesTuning {
  /** Body mass, kg. */
  mass: number
  /** Best-glide cruise airspeed, m/s. Lift exactly balances weight here. */
  cruise: number
  /** Stall airspeed, m/s. Lift collapses below this. */
  stall: number
  /** Never-exceed airspeed, m/s. Drag ramps hard beyond it. */
  vmax: number
  /** Best glide ratio (horizontal : vertical). */
  glide: number
  /** Wing-beat frequency at full power, Hz. */
  flapHz: number
  /** Peak flap thrust as a multiple of body weight. */
  power: number
  /** Bank response rate, 1/s (higher = snappier roll). */
  rollRate: number
  /** Maximum bank angle, radians. */
  maxBank: number
  /** Pitch-input authority: fractional lift change at full deflection. */
  pitchAuthority: number
  /** Camera follow stiffness, 1/s. Heavy birds lag; small birds snap. */
  camLag: number
  /** Head offset from the body origin: forward and up, meters. */
  headFwd: number
  headUp: number
}

/** Handling tables for every playable species. */
const TUNING: Record<SpeciesId, SpeciesTuning> = {
  // Heavy soarer: huge glide, stately roll, monstrous dive ceiling.
  'wedge-tailed-eagle': {
    mass: 4.2, cruise: 16, stall: 9.5, vmax: 62, glide: 15,
    flapHz: 2.6, power: 2.6, rollRate: 2.0, maxBank: 1.05,
    pitchAuthority: 1.5, camLag: 7, headFwd: 0.42, headUp: 0.16,
  },
  // Strong direct flier, showy but not twitchy.
  'sulphur-crested-cockatoo': {
    mass: 0.88, cruise: 13.5, stall: 8, vmax: 48, glide: 9,
    flapHz: 4.4, power: 3.2, rollRate: 3.2, maxBank: 1.2,
    pitchAuthority: 1.6, camLag: 10, headFwd: 0.3, headUp: 0.12,
  },
  // Darting and agile: rapid beats, razor roll, tight banks.
  'rainbow-lorikeet': {
    mass: 0.13, cruise: 12, stall: 6.5, vmax: 42, glide: 7,
    flapHz: 7.5, power: 4.2, rollRate: 5.0, maxBank: 1.35,
    pitchAuthority: 1.8, camLag: 14, headFwd: 0.18, headUp: 0.08,
  },
  // Playful mid-weight: quick beats, willing banks.
  galah: {
    mass: 0.33, cruise: 12.5, stall: 7, vmax: 44, glide: 8.5,
    flapHz: 5.2, power: 3.4, rollRate: 3.8, maxBank: 1.25,
    pitchAuthority: 1.7, camLag: 11, headFwd: 0.24, headUp: 0.1,
  },
  // Stout perch-hunter: burst flier, modest glide.
  'laughing-kookaburra': {
    mass: 0.34, cruise: 11.5, stall: 7, vmax: 40, glide: 6.5,
    flapHz: 4.8, power: 3.6, rollRate: 3.4, maxBank: 1.15,
    pitchAuthority: 1.6, camLag: 10, headFwd: 0.26, headUp: 0.1,
  },
}

/** An invisible thermal column: centre, radius, core updraft, and cap height. */
interface Thermal {
  x: number
  z: number
  /** Core radius, m. Updraft falls off as a Gaussian of this. */
  radius: number
  /** Peak vertical air velocity, m/s. */
  updraft: number
  /** Height (m ASL) where the thermal caps out. */
  top: number
}

/** Thermal field: strong columns over the arid eastern plateau, one at the valley head. */
const THERMALS: readonly Thermal[] = [
  { x: 2600, z: 300, radius: 260, updraft: 5.5, top: 1500 },
  { x: 3400, z: -900, radius: 300, updraft: 6.0, top: 1700 },
  { x: 4300, z: 700, radius: 240, updraft: 5.0, top: 1400 },
  { x: 2900, z: 1600, radius: 280, updraft: 4.5, top: 1300 },
  { x: 3800, z: 2600, radius: 260, updraft: 4.5, top: 1200 },
  { x: 1400, z: -2200, radius: 220, updraft: 4.0, top: 1600 },
]

/**
 * Photo-mode composition for `?shot=N` (1-based). The camera height is placed
 * relative to the terrain sampled at update time, so each shot sits exactly on
 * the landform it was composed for even if the terrain module retunes.
 */
interface Shot {
  x: number
  z: number
  /** Camera height above the local surface, m. */
  above: number
  /** Absolute minimum camera height (m ASL), e.g. to stay above the swell. */
  minY: number
  look: readonly [number, number, number]
  fov: number
}

/**
 * The four review compositions. Each leads the eye along a landform toward
 * light: dawn side-light on the range, glitter path on the sea, layered
 * ridges from the peak, raking gold across thermal country.
 */
const SHOTS: readonly Shot[] = [
  // 1 — Valley dawn: low over the river flats, the range towering in the NW,
  //     the river leading the eye into the foothills.
  { x: 320, z: 780, above: 75, minY: 55, look: [-3450, 900, -3550], fov: 60 },
  // 2 — The surf line: wave-top height, looking east along the beach so the
  //     breakers rake across frame into the sun's glitter path.
  { x: 1450, z: 3420, above: 22, minY: 20, look: [4600, 55, 4500], fov: 55 },
  // 3 — From the peak: the whole valley sweeping down to the sea, ridge lines
  //     stacked in aerial perspective.
  { x: -3380, z: -3320, above: 65, minY: 720, look: [600, -80, 4200], fov: 66 },
  // 4 — Plateau gold: low raking light across thermal country, the range a
  //     blue wall on the horizon.
  { x: 2760, z: 1020, above: 30, minY: 26, look: [-3300, 780, -3500], fov: 58 },
]

/**
 * Autofly waypoints [x, y, z]: spawn → dive to the river mouth → low along
 * the surf → climbing inland over the dunes → an arc through plateau thermal
 * country → across the valley → up and over the main peak → home down the
 * ridge line. Heights are minimums; the builder lifts them clear of terrain.
 *
 * The whole loop is flown deliberately low: parallax on screen scales with
 * speed over height, so the showcase hugs the terrain and lets the clearance
 * lift in {@link Flight.buildAutoflyPath} keep it honest over the ranges.
 */
const AUTOFLY_WAYPOINTS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 130, 1500],
  [300, 80, 2900],
  [950, 50, 3900],
  [2100, 40, 4250],
  [3200, 65, 4150],
  [4000, 190, 3100],
  [4300, 400, 1600],
  [3800, 540, -300],
  [3050, 640, -1050],
  [1500, 460, -1600],
  [-300, 350, -2600],
  [-2000, 720, -3200],
  [-3400, 1090, -3520],
  [-3650, 860, -1900],
  [-2400, 460, -450],
  [-1000, 220, 700],
]

/** Reusable scratch objects — the update path allocates nothing. */
const _vAir = new THREE.Vector3()
const _fwdAir = new THREE.Vector3()
const _perpUp = new THREE.Vector3()
const _rightH = new THREE.Vector3()
const _liftDir = new THREE.Vector3()
const _thrustDir = new THREE.Vector3()
const _accel = new THREE.Vector3()
const _wind = new THREE.Vector3()
const _normal = new THREE.Vector3()
const _bodyFwd = new THREE.Vector3()
const _bodyUp = new THREE.Vector3()
const _bodyRight = new THREE.Vector3()
const _camPos = new THREE.Vector3()
const _euler = new THREE.Euler(0, 0, 0, 'YXZ')
const _shakeEuler = new THREE.Euler(0, 0, 0, 'YXZ')
const _quatA = new THREE.Quaternion()
const _pathPoint = new THREE.Vector3()
const _pathTangent = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)

/**
 * The flight module: physics, input, environmental air, collisions, autofly,
 * photo presets, and the first-person camera. Owns and writes
 * {@link GameContext.flight} every frame.
 */
export class Flight implements GameModule {
  readonly name = 'flight'

  // ------------------------------------------------------------------ input
  private readonly keys = new Set<string>()
  private pointerLocked = false
  private mouseDX = 0
  private mouseDY = 0

  // ---------------------------------------------------------------- control
  /** Current bank angle, radians. Positive = right wing down. */
  private bank = 0
  /** Target bank angle from input, radians. */
  private bankTarget = 0
  /** Smoothed pitch input, -1 (nose down) .. 1 (pull up). */
  private pitchInput = 0
  /** Target pitch input from mouse/keys. */
  private pitchTarget = 0
  /** True while Shift holds the wings tucked. */
  private tucked = false

  // ---------------------------------------------------------------- tuning
  private tuning: SpeciesTuning = TUNING['wedge-tailed-eagle']
  private appliedSpecies: SpeciesId | null = null
  /** Body weight, N. */
  private weight = this.tuning.mass * G
  /** Lift coefficient scale: L = kLift · cl · v². */
  private kLift = 0
  /** Parasitic drag scale: D_p = kPara · v². */
  private kPara = 0
  /** Induced drag scale: D_i = kInd · L² / v². */
  private kInd = 0

  // ------------------------------------------------------------------ feel
  /** Stall buffet envelope, 0..1. */
  private buffet = 0
  /** Collision thud envelope, decays after a graze. */
  private thud = 0
  /**
   * Physical flap power, 0..1. Drives thrust and the camera surge. Kept
   * separate from the *visual* {@link FlightState.flapStrength}, which never
   * drops below a breathing soar baseline — the baseline must add no energy.
   */
  private flapPower = 0
  /** Smoothed downstroke surge, feeds the camera bob. */
  private surge = 0
  /** Smoothed camera orientation. */
  private readonly camQuat = new THREE.Quaternion()
  private camInitialized = false

  // ------------------------------------------------------------------- air
  private readonly noiseA: NoiseFunction2D = createNoise2D(() => 0.137)
  private readonly noiseB: NoiseFunction2D = createNoise2D(() => 0.571)
  private readonly noiseC: NoiseFunction2D = createNoise2D(() => 0.913)
  /** Current gust strength 0..1, drives subtle camera nudges. */
  private gust = 0

  // --------------------------------------------------------------- autofly
  private autoPath: THREE.CatmullRomCurve3 | null = null
  private autoLength = 1
  /** Distance travelled along the showcase spline, m. */
  private autoDist = 0
  private autoPrevYaw = 0
  private autoYawInit = false

  /** Wire up input listeners and the flight-start reset. */
  init(ctx: GameContext): void {
    const canvas = ctx.renderer.domElement
    canvas.addEventListener('click', () => {
      if (ctx.flying && !ctx.autofly && ctx.photoShot === null && !this.pointerLocked) {
        // Older engines return void; newer return a promise that can reject.
        try {
          void (canvas.requestPointerLock() as unknown as Promise<void> | undefined)?.catch(() => {})
        } catch {
          /* pointer lock unavailable — keyboard still flies the bird */
        }
      }
    })
    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === canvas
    })
    addEventListener('mousemove', (e) => {
      if (this.pointerLocked) {
        this.mouseDX += e.movementX
        this.mouseDY += e.movementY
      }
    })
    addEventListener('keydown', (e) => {
      this.keys.add(e.code)
      if (ctx.flying && (e.code === 'Space' || e.code.startsWith('Arrow'))) e.preventDefault()
    })
    addEventListener('keyup', (e) => this.keys.delete(e.code))
    addEventListener('blur', () => this.keys.clear())

    ctx.events.addEventListener('start-flight', () => this.resetFlight(ctx))
    this.applySpecies(ctx.species)
  }

  /** Per-frame dispatch across the four camera/physics modes. */
  update(dt: number, ctx: GameContext): void {
    if (ctx.photoShot !== null) {
      this.applyShot(dt, ctx)
      return
    }
    if (!ctx.flying) {
      if (!ctx.paused) this.updateMenuCamera(dt, ctx)
      return
    }
    if (ctx.paused) return
    if (this.appliedSpecies !== ctx.species) this.applySpecies(ctx.species)

    if (ctx.autofly) {
      this.updateAutofly(dt, ctx)
    } else {
      this.readInputs(dt)
      this.stepPhysics(dt, ctx)
    }
    this.composePose(ctx)
    this.updateCamera(dt, ctx)
  }

  // ================================================================== tuning

  /** Derive aerodynamic coefficients from the species' physical numbers. */
  private applySpecies(species: SpeciesId): void {
    const t = TUNING[species]
    this.tuning = t
    this.appliedSpecies = species
    this.weight = t.mass * G
    // Lift balances weight at cruise: L = kLift · v²  →  kLift = W / v_c².
    this.kLift = this.weight / (t.cruise * t.cruise)
    // At best glide, parasitic and induced drag are equal and sum to W / glide.
    this.kPara = this.weight / (2 * t.glide * t.cruise * t.cruise)
    // Induced drag = kInd · L² / v²; equals the parasitic half at cruise.
    this.kInd = (t.cruise * t.cruise) / (2 * t.glide * this.weight)
  }

  /** Reset the bird to the spawn point, cruising south over the valley. */
  private resetFlight(ctx: GameContext): void {
    this.applySpecies(ctx.species)
    const f = ctx.flight
    f.position.set(WORLD.spawn.x, WORLD.spawn.y, WORLD.spawn.z)
    f.velocity.set(0, 0, -this.tuning.cruise)
    f.speed = this.tuning.cruise
    f.flapPhase = 0
    f.flapStrength = 0
    f.banking = 0
    f.stalled = false
    this.bank = 0
    this.bankTarget = 0
    this.pitchInput = 0
    this.pitchTarget = 0
    this.buffet = 0
    this.thud = 0
    this.surge = 0
    this.flapPower = 0
    this.camInitialized = false
  }

  // =================================================================== input

  /** Consume mouse deltas and key state into smoothed bank/pitch targets. */
  private readInputs(dt: number): void {
    const t = this.tuning
    const dx = this.mouseDX
    const dy = this.mouseDY
    this.mouseDX = 0
    this.mouseDY = 0
    this.tucked = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight')

    const agility = t.rollRate / 3
    // Mouse X banks; mouse Y pitches (push forward to dive, pull back to climb).
    this.bankTarget += dx * 0.0032 * agility
    this.pitchTarget -= dy * 0.0028 * agility

    // Keyboard fallback.
    const left = this.keys.has('KeyA') || this.keys.has('ArrowLeft') ? 1 : 0
    const right = this.keys.has('KeyD') || this.keys.has('ArrowRight') ? 1 : 0
    const up = this.keys.has('KeyW') || this.keys.has('ArrowUp') ? 1 : 0
    const down = this.keys.has('KeyS') || this.keys.has('ArrowDown') ? 1 : 0
    this.bankTarget += (right - left) * t.rollRate * 0.9 * dt
    this.pitchTarget += (up - down) * 2.2 * dt

    // Birds naturally re-trim: bank and pitch ease back toward level flight.
    this.bankTarget -= this.bankTarget * Math.min(1, 0.6 * dt)
    this.pitchTarget -= this.pitchTarget * Math.min(1, 1.1 * dt)
    this.bankTarget = THREE.MathUtils.clamp(this.bankTarget, -t.maxBank, t.maxBank)
    this.pitchTarget = THREE.MathUtils.clamp(this.pitchTarget, -1, 1)

    this.bank = THREE.MathUtils.damp(this.bank, this.bankTarget, t.rollRate, dt)
    this.pitchInput = THREE.MathUtils.damp(this.pitchInput, this.pitchTarget, 6, dt)
  }

  // ================================================================= physics

  /** One aerodynamic step: forces from air-relative flow, then integration. */
  private stepPhysics(dt: number, ctx: GameContext): void {
    const t = this.tuning
    const f = ctx.flight
    const tuck = this.tucked
    const flapHeld = !tuck && (this.keys.has('Space') || this.keys.has('KeyF'))

    // --- flapping. Physical power ramps with the key. The *visual* wing
    //     drive never drops below a breathing soar baseline plus turbulence
    //     flex, so the first-person wings always articulate — a 1-2 Hz
    //     micro-oscillation in glides, the full beat under power. Thrust
    //     reads only the physical power; the baseline adds zero energy.
    const ramp = flapHeld ? 9 : 4
    this.flapPower = THREE.MathUtils.damp(this.flapPower, flapHeld ? 1 : 0, ramp, dt)
    // Glide flex = breathing floor + slow thermal breathing + fast turbulence
    // flutter + gust response + intermittent trim-flap bursts. Baseline range
    // ≈ 0.26..0.56, rising toward 0.85 through a burst: real gliding birds
    // punctuate soaring with a couple of honest re-trim beats every few
    // seconds, and those bursts are what a random capture should catch. All
    // of this drives the RIG only — thrust reads {@link flapPower} alone, so
    // the visual life adds zero energy to the physics.
    const burst = THREE.MathUtils.smoothstep(this.noiseB(ctx.time * 0.16, 12.5), 0.3, 0.75)
    const flex =
      WING_LIFE_MIN +
      0.1 * (this.noiseC(ctx.time * 0.6, 41.3) * 0.5 + 0.5) +
      0.06 * (this.noiseA(ctx.time * 2.3, 83.9) * 0.5 + 0.5) +
      0.12 * this.gust +
      0.45 * burst
    f.flapStrength = Math.max(this.flapPower, this.tucked ? 0.05 : Math.min(flex, WING_LIFE_MAX))
    // The beat quickens with real power, and a little through a visual trim
    // burst so those re-trim beats read at a working cadence, not slow motion.
    const beatHz = SOAR_HZ + (t.flapHz - SOAR_HZ) * Math.max(this.flapPower, 0.4 * burst)
    f.flapPhase = (f.flapPhase + dt * beatHz) % 1

    // --- environment: terrain sample once, then the local air vector.
    const ground = ctx.getTerrainHeight(f.position.x, f.position.z)
    const altitude = f.position.y - ground
    this.sampleWind(ctx, ground, altitude, _wind)

    // --- air-relative flow.
    _vAir.copy(f.velocity).sub(_wind)
    const vA = Math.max(_vAir.length(), 0.5)
    _fwdAir.copy(_vAir).divideScalar(vA)

    // --- lift coefficient from angle of attack (pitch input), tuck, stall, thin air.
    let cl = 1 + this.pitchInput * t.pitchAuthority
    if (tuck) cl *= 0.3
    const stallRatio = vA / t.stall
    const stalled = stallRatio < 1 && altitude > 3
    if (stallRatio < 1) cl *= stallRatio * stallRatio
    cl *= THREE.MathUtils.clamp(1 - (f.position.y - THIN_AIR_START) / 1200, 0.25, 1)

    // --- stall dynamics: the nose is forced down until airspeed returns.
    if (stalled) {
      const sink = 1 - stallRatio
      this.pitchTarget = Math.min(this.pitchTarget, -0.5 * sink)
      // Wings rock in the burble; bank authority softens.
      this.bank += this.noiseB(ctx.time * 6, 4.2) * 0.5 * sink * dt
    }

    // --- ground effect: a lift cushion and relaxed induced drag near the surface.
    const ge = Math.exp(-Math.max(altitude, 0) / 9)
    const liftBonus = 1 + 0.12 * ge

    // --- lift direction: perpendicular to the flow, tilted by bank to turn.
    _perpUp.copy(UP).addScaledVector(_fwdAir, -UP.dot(_fwdAir))
    if (_perpUp.lengthSq() < 1e-6) _perpUp.set(0, 1, 0)
    else _perpUp.normalize()
    _rightH.crossVectors(_fwdAir, UP)
    if (_rightH.lengthSq() < 1e-6) _rightH.set(1, 0, 0)
    else _rightH.normalize()
    const cosB = Math.cos(this.bank)
    const sinB = Math.sin(this.bank)
    _liftDir.copy(_perpUp).multiplyScalar(cosB).addScaledVector(_rightH, sinB)

    const lift = Math.min(this.kLift * cl * vA * vA * liftBonus, 3 * this.weight)

    // --- drag: parasitic + induced (relaxed in ground effect) + overspeed wall.
    let drag =
      this.kPara * (tuck ? 0.45 : 1) * vA * vA +
      this.kInd * (1 - 0.45 * ge) * (lift * lift) / (vA * vA)
    if (vA > t.vmax) drag += this.kPara * 3 * (vA - t.vmax) * (vA - t.vmax)

    // --- flap thrust: an impulse peaking on each downstroke, angled a little up
    //     so climbing under power feels like real wing loading, not a rocket.
    const beat = Math.max(0, Math.sin(f.flapPhase * Math.PI * 2))
    const thrust = t.power * this.weight * beat * beat * this.flapPower
    this.surge = THREE.MathUtils.damp(this.surge, beat * beat * this.flapPower, 12, dt)
    _thrustDir.copy(_fwdAir).addScaledVector(_perpUp, 0.3).normalize()

    // --- integrate (semi-implicit Euler).
    _accel
      .copy(_liftDir).multiplyScalar(lift / t.mass)
      .addScaledVector(_thrustDir, thrust / t.mass)
      .addScaledVector(_fwdAir, -drag / t.mass)
    _accel.y -= G
    this.applyBounds(f.position, _accel)
    f.velocity.addScaledVector(_accel, dt)
    f.position.addScaledVector(f.velocity, dt)

    // --- soft collision with terrain and water.
    this.resolveCollision(dt, ctx)

    // --- state out.
    f.speed = vA
    f.stalled = stalled
    // Turbulence rocks the wings: the multi-band {@link rockSignal} on the
    // normalized bank feeds the rig's roll-coupled asymmetric dihedral
    // (inside wing tucks, outside wing lifts). {@link ROCK_GAIN} sizes the
    // wander so even a dead-straight cruise carries ±0.3..0.55 of asymmetric
    // drive with a 1.3 Hz working flex — no still frame ever catches a
    // perfectly symmetric planar spread.
    f.banking = THREE.MathUtils.clamp(
      this.bank / t.maxBank + this.rockSignal(ctx.time) * ROCK_GAIN,
      -1,
      1,
    )
    f.altitude = f.position.y - ctx.getTerrainHeight(f.position.x, f.position.z)
    this.buffet = THREE.MathUtils.damp(this.buffet, stalled ? 1 : 0, 5, dt)
    this.thud = Math.max(0, this.thud - dt * 2.2)
  }

  /**
   * Local air velocity: prevailing sea breeze + slow simplex gusts + thermal
   * updrafts + ridge lift where the breeze runs up the windward slopes.
   */
  private sampleWind(ctx: GameContext, ground: number, altitude: number, out: THREE.Vector3): void {
    const time = ctx.time
    const p = ctx.flight.position
    const gx = this.noiseA(time * 0.11, 3.7)
    const gy = this.noiseB(time * 0.13, 7.1)
    const gz = this.noiseC(time * 0.09, 1.3)
    this.gust = Math.min(1, Math.abs(gx) + Math.abs(gz))
    out.set(WIND_BASE_X + gx * 2.4, gy * 0.9, WIND_BASE_Z + gz * 2.4)

    // Thermals: Gaussian columns that fade in above the surface and cap on top.
    for (let i = 0; i < THERMALS.length; i++) {
      const th = THERMALS[i]
      const dx = p.x - th.x
      const dz = p.z - th.z
      const r2 = dx * dx + dz * dz
      const rr = th.radius * th.radius
      if (r2 > rr * 9) continue
      const fadeIn = THREE.MathUtils.clamp((p.y - ground) / 60, 0, 1)
      const fadeOut = THREE.MathUtils.clamp((th.top - p.y) / 300, 0, 1)
      out.y += th.updraft * Math.exp(-r2 / rr) * fadeIn * fadeOut
    }

    // Ridge lift: vertical air = horizontal wind dotted with the terrain
    // gradient, strongest hugging the slope and fading ~130 m above it.
    const dhdx = (ctx.getTerrainHeight(p.x + 20, p.z) - ctx.getTerrainHeight(p.x - 20, p.z)) / 40
    const dhdz = (ctx.getTerrainHeight(p.x, p.z + 20) - ctx.getTerrainHeight(p.x, p.z - 20)) / 40
    const slopeWind = out.x * dhdx + out.z * dhdz
    const hug = Math.exp(-Math.max(altitude, 0) / 130)
    out.y += (slopeWind > 0 ? slopeWind * 1.6 : slopeWind * 0.4) * hug
    out.y = THREE.MathUtils.clamp(out.y, -6, 9)
  }

  /**
   * Turbulence wing-rock: the raw asymmetric-dihedral drive mixed into
   * {@link FlightState.banking} (after {@link ROCK_GAIN}). Four incommensurate
   * bands, so the sum almost never nulls and always carries visible motion:
   * - a slow simplex wander, gust-scaled — the moving air itself,
   * - a 0.075 Hz sine — long, lazy weight shifts,
   * - a 1.3 Hz sine — the soaring-band flex the art brief demands: each wing
   *   trims against the other at a visible working rate,
   * - fast simplex flutter, gust-scaled — chatter over rough air.
   * Peak ≈ ±2.0 in a full gust, typical ≈ ±0.9. Shared by free flight,
   * autofly, and photo mode so every capture path carries identical life.
   * Pure function of time + the {@link gust} field: zero allocations.
   */
  private rockSignal(time: number): number {
    return (
      this.noiseB(time * 0.8, 33.7) * (0.5 + 0.5 * this.gust) +
      Math.sin(time * 0.47 + 1.3) * 0.35 +
      Math.sin(time * 8.2 + 0.7) * 0.3 +
      this.noiseC(time * 1.9, 54.2) * (0.15 + 0.2 * this.gust)
    )
  }

  /** Steer gently back toward the map when the bird nears the world edge. */
  private applyBounds(p: THREE.Vector3, accel: THREE.Vector3): void {
    if (p.x > SOFT_BOUND) accel.x -= (p.x - SOFT_BOUND) * 0.05
    else if (p.x < -SOFT_BOUND) accel.x += (-SOFT_BOUND - p.x) * 0.05
    if (p.z > SOFT_BOUND) accel.z -= (p.z - SOFT_BOUND) * 0.05
    else if (p.z < -SOFT_BOUND) accel.z += (-SOFT_BOUND - p.z) * 0.05
  }

  /**
   * Soft collision: kill the into-surface velocity component, slide along the
   * local surface with friction, and record a thud for the camera. Water uses
   * a flat normal and heavier drag. The bird grazes — it never teleports.
   */
  private resolveCollision(dt: number, ctx: GameContext): void {
    const f = ctx.flight
    const ground = ctx.getTerrainHeight(f.position.x, f.position.z)
    const overWater = ground < WORLD.seaLevel
    const floor = Math.max(ground, WORLD.seaLevel) + 0.6
    if (f.position.y >= floor) return

    if (overWater) {
      _normal.set(0, 1, 0)
    } else {
      const dhdx =
        (ctx.getTerrainHeight(f.position.x + 4, f.position.z) -
          ctx.getTerrainHeight(f.position.x - 4, f.position.z)) / 8
      const dhdz =
        (ctx.getTerrainHeight(f.position.x, f.position.z + 4) -
          ctx.getTerrainHeight(f.position.x, f.position.z - 4)) / 8
      _normal.set(-dhdx, 1, -dhdz).normalize()
    }

    f.position.y = floor
    const vn = f.velocity.dot(_normal)
    if (vn < 0) {
      f.velocity.addScaledVector(_normal, -vn * 1.1)
      this.thud = Math.min(1, this.thud + Math.min(1, -vn * 0.06))
    }
    const friction = overWater ? 2.8 : 1.6
    f.velocity.multiplyScalar(Math.max(0, 1 - friction * dt))
  }

  // ==================================================================== pose

  /** Build the body orientation from the velocity track, bank, and posture. */
  private composePose(ctx: GameContext): void {
    const f = ctx.flight
    const speed = Math.max(f.velocity.length(), 0.5)

    const yaw = speed > 1.5 ? Math.atan2(-f.velocity.x, -f.velocity.z) : _euler.y
    const track = Math.asin(THREE.MathUtils.clamp(f.velocity.y / speed, -1, 1))
    let pitch = track + this.pitchInput * 0.22
    if (this.tucked && !ctx.autofly) pitch -= 0.3
    pitch = THREE.MathUtils.clamp(pitch, -1.25, 1.25)

    _euler.set(pitch, yaw, -this.bank)
    f.quaternion.setFromEuler(_euler)
  }

  // ================================================================== camera

  /**
   * First-person camera at the head: flap-synced bob with a downstroke surge,
   * banked horizon, gust nudges, dive/stall/thud shake, and a speed-scaled
   * {@link FOV_MIN}→{@link FOV_MAX} field of view.
   */
  private updateCamera(dt: number, ctx: GameContext): void {
    const t = this.tuning
    const f = ctx.flight
    const cam = ctx.camera

    _bodyFwd.set(0, 0, -1).applyQuaternion(f.quaternion)
    _bodyUp.set(0, 1, 0).applyQuaternion(f.quaternion)
    _bodyRight.set(1, 0, 0).applyQuaternion(f.quaternion)

    // Head position: forward of the body, riding the wing beat. The head leads
    // slightly into the turn the way a bird looks through its bank.
    const beat = Math.sin(f.flapPhase * Math.PI * 2 + 0.6)
    _camPos
      .copy(f.position)
      .addScaledVector(_bodyFwd, t.headFwd + this.surge * 0.03)
      .addScaledVector(_bodyUp, t.headUp + beat * 0.055 * f.flapStrength)
      .addScaledVector(_bodyRight, f.banking * 0.04)

    // Shake: fast dives, stall buffet, collision thuds; slow gust nudges.
    const diveShake =
      THREE.MathUtils.smoothstep(f.speed, t.vmax * 0.68, t.vmax) * (this.tucked ? 0.014 : 0.008)
    const shake = diveShake + this.buffet * 0.013 + this.thud * 0.035
    const tt = ctx.time
    const nudge = 0.008 * this.gust
    _shakeEuler.set(
      this.noiseA(tt * 11, 17.3) * shake + this.noiseA(tt * 0.5, 27.7) * nudge,
      this.noiseB(tt * 10, 51.9) * shake * 0.7 + this.noiseB(tt * 0.4, 63.1) * nudge,
      this.noiseC(tt * 12, 39.2) * shake + this.noiseC(tt * 0.6, 88.4) * nudge * 1.5,
    )
    _quatA.setFromEuler(_shakeEuler).premultiply(f.quaternion)

    // Smoothed orientation: heavy birds carry momentum, small birds snap to it.
    if (!this.camInitialized) {
      this.camQuat.copy(_quatA)
      cam.position.copy(_camPos)
      this.camInitialized = true
    }
    this.camQuat.slerp(_quatA, 1 - Math.exp(-t.camLag * dt))
    cam.quaternion.copy(this.camQuat)
    cam.position.copy(_camPos)

    // Speed-scaled field of view: FOV_MIN at a stroll, FOV_MAX flat out.
    const speedN = THREE.MathUtils.smoothstep(f.speed, t.cruise * 0.8, t.vmax)
    cam.fov = THREE.MathUtils.damp(cam.fov, FOV_MIN + (FOV_MAX - FOV_MIN) * speedN, 3, dt)
    cam.updateProjectionMatrix()
  }

  /** Slow aerial drift over the valley while the menu is up. */
  private updateMenuCamera(dt: number, ctx: GameContext): void {
    const a = ctx.time * 0.018
    const x = WORLD.spawn.x + Math.cos(a) * 850
    const z = WORLD.spawn.z + Math.sin(a) * 850
    const ground = ctx.getTerrainHeight(x, z)
    const y = Math.max(ground + 150, 220) + Math.sin(ctx.time * 0.11) * 18
    ctx.camera.position.set(x, y, z)
    ctx.camera.lookAt(-1500, 420, -1800)
    ctx.camera.fov = THREE.MathUtils.damp(ctx.camera.fov, 66, 2, dt)
    ctx.camera.updateProjectionMatrix()
    // Park the flight state under the camera so LOD systems stay centred.
    ctx.flight.position.copy(ctx.camera.position)
    ctx.flight.quaternion.copy(ctx.camera.quaternion)
    ctx.flight.altitude = y - ground
  }

  // ================================================================= autofly

  /**
   * Scripted showcase loop, flown at a variable pace: fast on the descents
   * (the dive to the coast reads as a dive), slower and flappier on the
   * climbs. The bird banks into every turn the path demands, so the whole
   * loop looks hand-flown rather than railed.
   */
  private updateAutofly(dt: number, ctx: GameContext): void {
    if (!this.autoPath) this.buildAutoflyPath(ctx)
    const path = this.autoPath as THREE.CatmullRomCurve3
    const f = ctx.flight
    const t = this.tuning

    // Sample the tangent first so the pace can respond to the slope ahead.
    // Showcase pace: fast on the flats so low ground streams through the frame
    // within seconds, near-vne on the dives, still quick on the climbs.
    let u = (this.autoDist / this.autoLength) % 1
    path.getTangentAt(u, _pathTangent)
    const speed = THREE.MathUtils.clamp(44 - _pathTangent.y * 50, 32, 68)
    this.autoDist += speed * dt
    u = (this.autoDist / this.autoLength) % 1

    path.getPointAt(u, _pathPoint)
    path.getTangentAt(u, _pathTangent)

    // Keep a safety margin above the rendered surface.
    const ground = ctx.getTerrainHeight(_pathPoint.x, _pathPoint.z)
    const floor = Math.max(ground, WORLD.seaLevel) + 12
    if (_pathPoint.y < floor) _pathPoint.y = floor

    // Flap-glide rhythm: full power on the climbs, frequent beat bursts over
    // an always-working baseline on the cruises, clean glides only on real
    // descents. Any still frame catches the wings mid-cycle, never parked.
    const climbing = _pathTangent.y > 0.015
    const diving = _pathTangent.y < -0.05
    const burst = this.noiseA(ctx.time * 0.35, 99.1) > -0.55
    // Even the diving glide keeps the soar-baseline micro-oscillation alive;
    // the floors sit on {@link WING_LIFE_MIN} so no capture reads as parked.
    const flapTarget = climbing ? 1 : diving ? 0.3 : burst ? 0.95 : 0.55
    f.flapStrength = THREE.MathUtils.damp(f.flapStrength, flapTarget, 3.5, dt)
    const drive = THREE.MathUtils.clamp((f.flapStrength - 0.15) / 0.85, 0, 1)
    f.flapPhase = (f.flapPhase + dt * (SOAR_HZ + (t.flapHz - SOAR_HZ) * drive)) % 1
    const beat = Math.sin(f.flapPhase * Math.PI * 2)
    this.surge = THREE.MathUtils.damp(this.surge, Math.max(0, beat) * f.flapStrength, 12, dt)

    // The whole bird rides each downstroke: a ~0.45 m vertical bob plus a
    // nose rock (via pitchInput → composePose) so the horizon itself beats.
    f.position.copy(_pathPoint)
    f.position.y += beat * 0.45 * f.flapStrength
    f.velocity.copy(_pathTangent).multiplyScalar(speed)
    f.speed = speed
    f.altitude = f.position.y - ground
    this.pitchInput = beat * 0.35 * f.flapStrength

    // Coordinated bank from the turn rate the path demands, amplified for the
    // showcase, plus a slow sine-and-noise carve (~10-15°) so straights never
    // freeze the horizon — the camera is always rolling through something.
    const yaw = Math.atan2(-f.velocity.x, -f.velocity.z)
    if (!this.autoYawInit) {
      this.autoPrevYaw = yaw
      this.autoYawInit = true
    }
    let dYaw = yaw - this.autoPrevYaw
    if (dYaw > Math.PI) dYaw -= Math.PI * 2
    else if (dYaw < -Math.PI) dYaw += Math.PI * 2
    this.autoPrevYaw = yaw
    const lateral = (-dYaw / Math.max(dt, 1e-4)) * speed
    // Two incommensurate sines plus noise: the sum almost never nulls out, so
    // no still frame catches a dead-level horizon on a straight.
    const carve =
      Math.sin(ctx.time * 0.55) * 0.17 +
      Math.sin(ctx.time * 0.23 + 2.1) * 0.12 +
      this.noiseB(ctx.time * 0.3, 7.7) * 0.12
    const bankTarget = THREE.MathUtils.clamp(
      Math.atan2(lateral, G) * 1.5 + carve,
      -t.maxBank,
      t.maxBank,
    )
    this.bank = THREE.MathUtils.damp(this.bank, bankTarget, 3, dt)
    // Same turbulence rock as free flight: keeps the wing dihedral asymmetric
    // and working at 1.3 Hz in every capture, even on the straights between
    // carves. Gust is pinned first so the shared signal is fully determined.
    this.gust = 0.5
    f.banking = THREE.MathUtils.clamp(
      this.bank / t.maxBank + this.rockSignal(ctx.time) * ROCK_GAIN,
      -1,
      1,
    )
    f.stalled = false
    this.tucked = false
    this.buffet = 0
  }

  /** Lay the showcase spline once the terrain sampler is live, lifted clear of the surface. */
  private buildAutoflyPath(ctx: GameContext): void {
    const points = AUTOFLY_WAYPOINTS.map(([x, y, z]) => {
      // Extra clearance in the ranges, where slopes move fast under the path;
      // elsewhere stay low — parallax on screen scales with speed over height.
      const clearance = x < -1500 && z < -1500 ? 110 : 40
      return new THREE.Vector3(x, Math.max(y, ctx.getTerrainHeight(x, z) + clearance), z)
    })
    this.autoPath = new THREE.CatmullRomCurve3(points, true, 'centripetal', 0.6)
    this.autoPath.arcLengthDivisions = 600
    this.autoLength = this.autoPath.getLength()
    this.autoDist = 0
    this.autoYawInit = false
  }

  // ============================================================== photo mode

  /**
   * Photo mode: place the camera at the requested preset (terrain-aware every
   * frame), freeze flight physics, and park the flight state under the camera
   * so LOD, audio, and effects stay coherent while clouds and water continue
   * to animate.
   */
  private applyShot(dt: number, ctx: GameContext): void {
    const shot = SHOTS[((ctx.photoShot ?? 1) - 1 + SHOTS.length * 8) % SHOTS.length]
    const ground = ctx.getTerrainHeight(shot.x, shot.z)
    const y = Math.max(Math.max(ground, WORLD.seaLevel) + shot.above, shot.minY)
    const cam = ctx.camera
    cam.position.set(shot.x, y, shot.z)
    cam.lookAt(shot.look[0], shot.look[1], shot.look[2])
    cam.fov = THREE.MathUtils.damp(cam.fov, shot.fov, 8, dt)
    cam.updateProjectionMatrix()

    const f = ctx.flight
    f.position.copy(cam.position)
    f.quaternion.copy(cam.quaternion)
    f.velocity.set(0, 0, 0)
    // Airspeed stays at cruise: the rig's feather flutter and the wind bed
    // keep working while the camera is parked.
    f.speed = this.tuning.cruise
    f.altitude = y - ground
    // The bird soars in place with real working wings: a breathing baseline
    // that keeps the shoulder swinging ~±15-25°, slow flap bursts every few
    // seconds that carry the joints through a near-full beat, and a wide
    // banking wander driving the rig's asymmetric dihedral. Any capture, at
    // any moment, catches a distinct mid-stroke, left/right-asymmetric pose —
    // never a frozen symmetric spread.
    const burst = THREE.MathUtils.smoothstep(this.noiseC(ctx.time * 0.2, 5.1), 0.2, 0.65)
    f.flapStrength = THREE.MathUtils.damp(
      f.flapStrength,
      Math.min(
        0.38 + 0.14 * this.noiseA(ctx.time * 0.4, 61.7) + 0.5 * burst,
        WING_LIFE_MAX,
      ),
      3,
      dt,
    )
    // Beat cadence follows the burst: soar trim at rest, a working flap
    // through the burst, so the motion blur direction differs shot to shot.
    f.flapPhase =
      (f.flapPhase + dt * (SOAR_HZ + (this.tuning.flapHz - SOAR_HZ) * 0.5 * burst)) % 1
    // Banking = a slow held-bank wander plus the shared multi-band rock, so a
    // photo capture carries both a definite lean AND the 1.3 Hz left/right
    // flex — the two wings never mirror each other in any frame.
    this.gust = 0.5
    f.banking = THREE.MathUtils.clamp(
      this.noiseB(ctx.time * 0.22, 9.4) * 0.7 +
        Math.sin(ctx.time * 0.31) * 0.3 +
        this.rockSignal(ctx.time) * ROCK_GAIN,
      -1,
      1,
    )
    f.stalled = false
  }
}
