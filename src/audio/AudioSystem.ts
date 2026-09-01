/**
 * Aviary audio system — a fully synthesized WebAudio soundscape. No samples.
 *
 * Signal architecture (built once, on the first user gesture):
 *
 * ```
 *   wind banks ─┐
 *   flap voices ├─→ duck ─→ compressor ─→ master ─→ destination
 *   ambience  ──┤     ↑
 *   calls ──────┤     │
 *      └─(send)─→ convolver "outdoor space" ─→ return ─┘
 * ```
 *
 * Layers:
 * - **Wind** — four decorrelated noise banks: a low rumble, a stereo mid
 *   "body" pair whose pan leans into the bank angle, a high-Q whistle that
 *   opens up in fast dives, and a turbulence band that flutters with hard
 *   banking and flapping. Cutoffs and gains track airspeed and altitude;
 *   slow simplex gust swells breathe through the whole bed. A stall buffet
 *   tremolo shakes the rumble while the bird is below stall speed.
 * - **Wing flaps** — pooled two-part voices (a swept band-pass feather
 *   whoosh plus a low air thump), envelope-triggered when
 *   {@link FlightState.flapPhase} crosses the downstroke, weight-matched to
 *   the selected species.
 * - **Ambience** — a two-band surf bed that fades in near the southern
 *   coast, a dual-rate cicada shimmer over dry inland country by day, and
 *   sparse distant calls: soft whistles/warbles plus a rare formant-swept
 *   kookaburra laugh, both washed through a procedurally generated
 *   convolver impulse so they read as far away outdoors.
 * - **Doppler** — a subtle pitch rise on the looping wind sources
 *   (playbackRate) in fast dives.
 *
 * The mix runs through a gentle master compressor; a duck gain pulls
 * everything down while the menu is up or the game is paused. The
 * AudioContext is created and resumed only on the first user gesture
 * (click or keydown), per browser autoplay policy. Nothing beyond the
 * {@link GameModule} contract is exposed.
 */
import { createNoise2D } from 'simplex-noise'
import type { GameContext, GameModule, SpeciesId } from '../core/GameState'

// ---------------------------------------------------------------------------
// Species voicing
// ---------------------------------------------------------------------------

/**
 * Per-species wing-flap voicing. Heavier birds get lower, longer, thumpier
 * whooshes; small parrots get short, high feather flicks.
 */
interface FlapProfile {
  /** Band-pass center frequency at the start of the whoosh sweep, Hz. */
  freq: number
  /** Peak whoosh gain at full flap strength. */
  gain: number
  /** Whoosh envelope duration, seconds. */
  dur: number
  /** Whoosh filter Q — higher reads as a tighter, faster wing. */
  q: number
  /** Low air-thump gain relative to the whoosh (mass of the downstroke). */
  thump: number
}

/** Flap voicing table for every playable species. */
const FLAP_PROFILES: Record<SpeciesId, FlapProfile> = {
  'wedge-tailed-eagle': { freq: 310, gain: 0.8, dur: 0.34, q: 1.0, thump: 0.9 },
  'sulphur-crested-cockatoo': { freq: 540, gain: 0.58, dur: 0.2, q: 1.4, thump: 0.45 },
  galah: { freq: 660, gain: 0.48, dur: 0.16, q: 1.5, thump: 0.32 },
  'rainbow-lorikeet': { freq: 1350, gain: 0.3, dur: 0.09, q: 2.2, thump: 0.1 },
  'laughing-kookaburra': { freq: 560, gain: 0.55, dur: 0.18, q: 1.3, thump: 0.5 },
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

/** Frame-rate-independent exponential smoothing of `current` toward `target`. */
function smooth(current: number, target: number, dt: number, tau: number): number {
  return current + (target - current) * (1 - Math.exp(-dt / tau))
}

/** Hermite smoothstep of `x` remapped from [e0, e1] to [0, 1]. */
function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)))
  return t * t * (3 - 2 * t)
}

/** Clamp `x` to [lo, hi]. */
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x
}

// ---------------------------------------------------------------------------
// Node graph record
// ---------------------------------------------------------------------------

/** One pooled flap voice: always-running noise gated open by envelopes. */
interface FlapVoice {
  /** Feather-whoosh band-pass (frequency swept per trigger). */
  bp: BiquadFilterNode
  /** Whoosh envelope gain. */
  bpGain: GainNode
  /** Low air-thump envelope gain (behind a shared low-pass). */
  lpGain: GainNode
}

/** Every node the system owns. Built once after the first user gesture. */
interface AudioGraph {
  duck: GainNode
  /** Wet send into the outdoor-space convolver (distant calls, surf). */
  spaceSend: GainNode
  // Wind bed -----------------------------------------------------------------
  windRumbleFilter: BiquadFilterNode
  windRumbleGain: GainNode
  /** Stall-buffet tremolo stage inline with the rumble. */
  buffetGain: GainNode
  windBodyFilterL: BiquadFilterNode
  windBodyGainL: GainNode
  windBodyPanL: StereoPannerNode
  windBodyFilterR: BiquadFilterNode
  windBodyGainR: GainNode
  windBodyPanR: StereoPannerNode
  whistleFilter: BiquadFilterNode
  whistleGain: GainNode
  turbFilter: BiquadFilterNode
  turbGain: GainNode
  /** Looping noise sources feeding the wind bed (doppler via playbackRate). */
  windSources: AudioBufferSourceNode[]
  // Flaps ---------------------------------------------------------------------
  flapVoices: FlapVoice[]
  // Ambience ------------------------------------------------------------------
  surfLowGain: GainNode
  surfLowFilter: BiquadFilterNode
  surfWashGain: GainNode
  cicadaGain: GainNode
  cicadaFilterA: BiquadFilterNode
  // Calls ---------------------------------------------------------------------
  kookaOscA: OscillatorNode
  kookaOscB: OscillatorNode
  kookaFormant1: BiquadFilterNode
  kookaFormant2: BiquadFilterNode
  kookaGain: GainNode
  kookaPan: StereoPannerNode
  chirpOsc: OscillatorNode
  chirpGain: GainNode
  chirpPan: StereoPannerNode
}

// ---------------------------------------------------------------------------
// The module
// ---------------------------------------------------------------------------

/**
 * Synthesized soundscape module for Aviary. Owns the entire WebAudio graph
 * and exposes nothing beyond the {@link GameModule} contract.
 */
export class AudioSystem implements GameModule {
  readonly name = 'audio'

  private ac: AudioContext | null = null
  private g: AudioGraph | null = null
  private readonly noise2D = createNoise2D()

  // Smoothed control values, kept across frames (zero per-frame allocation).
  private windLow = 0
  private windBody = 0
  private windWhistle = 0
  private windTurb = 0
  private windCutoff = 180
  private bodyCutoff = 480
  private whistleCutoff = 2600
  private doppler = 1
  private duckLevel = 0
  private surfLevel = 0
  private cicadaLevel = 0
  private buffet = 0

  // Flap-phase edge detection state.
  private prevFlapPhase = 0
  private flapAcc = 0
  private lastFlapIndex = -1
  private nextVoice = 0

  // Call scheduling in game-time seconds, so calls pause with the game.
  private nextChirpAt = 10
  private nextKookaAt = 35

  /** Where the downstroke whoosh fires inside the 0..1 wing-beat cycle. */
  private static readonly DOWNSTROKE = 0.18

  /**
   * Registers first-gesture listeners. The AudioContext is not created (or
   * resumed) until the user clicks or presses a key, satisfying autoplay
   * policy. Listeners stay attached until the context is actually running,
   * so a gesture that lands while the tab is throttled still recovers.
   */
  init(_ctx: GameContext): void {
    const onGesture = (): void => {
      if (!this.ac) {
        this.ac = new AudioContext()
        this.g = this.buildGraph(this.ac)
      }
      if (this.ac.state === 'suspended') void this.ac.resume()
      if (this.ac.state === 'running') {
        removeEventListener('click', onGesture)
        removeEventListener('keydown', onGesture)
      }
    }
    addEventListener('click', onGesture)
    addEventListener('keydown', onGesture)
  }

  /** Drives every layer from the shared flight and world state. */
  update(dt: number, ctx: GameContext): void {
    const ac = this.ac
    const g = this.g
    if (!ac || !g || ac.state !== 'running') return

    const f = ctx.flight
    const t = ctx.time

    // ---- Global dynamics ---------------------------------------------------
    const speedN = clamp(f.speed / 45, 0, 1)
    const altN = clamp(f.altitude / 600, 0, 1)
    // Dive factor: fast AND descending. Drives the whistle band and doppler.
    const diveN = clamp(-f.velocity.y / 26, 0, 1) * smoothstep(0.35, 0.9, speedN)
    // Slow gust swells: two incommensurate simplex bands plus altitude bias.
    const gust =
      0.68 +
      0.22 * this.noise2D(t * 0.06, 0.0) +
      0.13 * this.noise2D(t * 0.21, 7.3) +
      0.08 * altN

    // ---- Menu / pause ducking ----------------------------------------------
    const duckTarget = ctx.flying && !ctx.paused ? 1 : 0.1
    this.duckLevel = smooth(this.duckLevel, duckTarget, dt, 0.35)
    g.duck.gain.value = this.duckLevel

    // ---- Wind bed ------------------------------------------------------------
    const exposure = 0.22 + 0.2 * altN // higher air is more exposed
    this.windLow = smooth(this.windLow, (0.16 + 0.48 * speedN + 0.3 * exposure) * gust, dt, 0.12)
    this.windBody = smooth(this.windBody, (0.05 + 0.55 * speedN) * gust, dt, 0.1)
    this.windWhistle = smooth(this.windWhistle, 0.02 * speedN + 0.32 * diveN * diveN, dt, 0.08)
    this.windCutoff = smooth(this.windCutoff, 140 + 640 * speedN, dt, 0.2)
    this.bodyCutoff = smooth(this.bodyCutoff, 360 + 1100 * speedN, dt, 0.2)
    this.whistleCutoff = smooth(this.whistleCutoff, 2400 + 2400 * diveN + 700 * speedN, dt, 0.15)

    g.windRumbleGain.gain.value = this.windLow
    g.windRumbleFilter.frequency.value = this.windCutoff
    g.windBodyGainL.gain.value = this.windBody
    g.windBodyGainR.gain.value = this.windBody * 0.92
    g.windBodyFilterL.frequency.value = this.bodyCutoff
    g.windBodyFilterR.frequency.value = this.bodyCutoff * 1.14
    g.whistleGain.gain.value = this.windWhistle
    g.whistleFilter.frequency.value = this.whistleCutoff

    // The wind body leans into the bank: air rushes on the low wing's side.
    const bank = clamp(f.banking, -1, 1)
    g.windBodyPanL.pan.value = -0.35 + 0.28 * bank
    g.windBodyPanR.pan.value = 0.35 + 0.28 * bank

    // Turbulence band: flutters with hard banking, flapping, and shed air.
    const flutter = 0.55 + 0.45 * this.noise2D(t * 2.6, 3.1)
    const turbTarget = (0.24 * Math.abs(bank) + 0.12 * f.flapStrength) * speedN * flutter
    this.windTurb = smooth(this.windTurb, turbTarget, dt, 0.07)
    g.turbGain.gain.value = this.windTurb
    g.turbFilter.frequency.value = 700 + 900 * speedN

    // Stall buffet: a ragged low shake while below stall speed.
    this.buffet = smooth(this.buffet, f.stalled ? 1 : 0, dt, 0.15)
    g.buffetGain.gain.value =
      1 - this.buffet * 0.45 * (0.5 + 0.5 * Math.sin(t * 52 + 2.2 * this.noise2D(t * 1.7, 11)))

    // Doppler-style pitch rise in fast dives via loop playback rate.
    this.doppler = smooth(this.doppler, 1 + 0.32 * diveN + 0.05 * speedN, dt, 0.25)
    for (let i = 0; i < g.windSources.length; i++) {
      g.windSources[i].playbackRate.value = this.doppler * (1 + i * 0.013)
    }

    // ---- Wing flaps ----------------------------------------------------------
    this.detectFlap(ac, g, ctx)

    // ---- Location ambience ---------------------------------------------------
    const px = f.position.x
    const pz = f.position.z
    const day = smoothstep(0.03, 0.25, ctx.sunDirection.y)

    // Surf: the ocean lies south (z > ~3000). Fade by proximity and height.
    const coast = smoothstep(1400, 3200, pz) * (1 - 0.75 * smoothstep(120, 700, f.altitude))
    this.surfLevel = smooth(this.surfLevel, coast * 0.5, dt, 0.8)
    // Wave sets: two offset simplex phases so swell and wash interleave.
    const swell = 0.5 + 0.5 * Math.max(0, this.noise2D(t * 0.1, 21.5))
    const wash = Math.max(0, this.noise2D(t * 0.16, 40.2))
    g.surfLowGain.gain.value = this.surfLevel * (0.5 + 0.5 * swell)
    g.surfLowFilter.frequency.value = 360 + 300 * swell
    g.surfWashGain.gain.value = this.surfLevel * 0.55 * wash * wash

    // Cicadas: inland (away from the coast), low, dry, daytime only.
    const inland = 1 - smoothstep(600, 2400, pz)
    const east = 0.55 + 0.45 * smoothstep(-500, 2500, px) // drier country east
    const lowAlt = 1 - smoothstep(60, 340, f.altitude)
    this.cicadaLevel = smooth(this.cicadaLevel, inland * east * lowAlt * day * 0.085, dt, 1.2)
    g.cicadaGain.gain.value = this.cicadaLevel
    // The chorus drifts slowly across the paddocks.
    g.cicadaFilterA.frequency.value = 5900 + 500 * this.noise2D(t * 0.05, 60)

    // ---- Distant calls -------------------------------------------------------
    if (ctx.flying && !ctx.paused) {
      if (t >= this.nextChirpAt) {
        this.triggerChirp(ac, g)
        this.nextChirpAt = t + 9 + Math.random() * 18
      }
      if (t >= this.nextKookaAt) {
        this.triggerKookaburra(ac, g)
        this.nextKookaAt = t + 55 + Math.random() * 85
      }
    }
  }

  // -------------------------------------------------------------------------
  // Graph construction
  // -------------------------------------------------------------------------

  /** Builds the full node graph. Called once, after the first user gesture. */
  private buildGraph(ac: AudioContext): AudioGraph {
    const master = ac.createGain()
    master.gain.value = 0
    // Fade the whole mix in over ~a second to avoid a boot pop.
    master.gain.setTargetAtTime(0.82, ac.currentTime + 0.05, 0.4)
    master.connect(ac.destination)

    const compressor = ac.createDynamicsCompressor()
    compressor.threshold.value = -20
    compressor.knee.value = 22
    compressor.ratio.value = 3
    compressor.attack.value = 0.012
    compressor.release.value = 0.28
    compressor.connect(master)

    const duck = ac.createGain()
    duck.gain.value = 0
    duck.connect(compressor)

    // Outdoor-space convolver: a procedurally generated stereo decay that
    // makes distant calls sit far away instead of inside the listener's head.
    const convolver = ac.createConvolver()
    convolver.buffer = this.makeSpaceImpulse(ac, 2.2)
    const spaceSend = ac.createGain()
    spaceSend.gain.value = 1
    const spaceReturn = ac.createGain()
    spaceReturn.gain.value = 0.5
    spaceSend.connect(convolver)
    convolver.connect(spaceReturn)
    spaceReturn.connect(duck)

    const pink = this.makePinkNoiseBuffer(ac, 4)
    const white = this.makeWhiteNoiseBuffer(ac, 3)

    const windSources: AudioBufferSourceNode[] = []
    /** Starts a looping buffer source at an offset (decorrelates the banks). */
    const startLoop = (
      buffer: AudioBuffer,
      dest: AudioNode,
      offset: number,
      trackDoppler = false,
    ): void => {
      const src = ac.createBufferSource()
      src.buffer = buffer
      src.loop = true
      src.connect(dest)
      src.start(0, offset)
      if (trackDoppler) windSources.push(src)
    }

    // ---- Wind: rumble (center, through the stall-buffet tremolo stage).
    const windRumbleFilter = ac.createBiquadFilter()
    windRumbleFilter.type = 'lowpass'
    windRumbleFilter.frequency.value = 180
    windRumbleFilter.Q.value = 0.5
    const windRumbleGain = ac.createGain()
    windRumbleGain.gain.value = 0
    const buffetGain = ac.createGain()
    buffetGain.gain.value = 1
    windRumbleFilter.connect(windRumbleGain)
    windRumbleGain.connect(buffetGain)
    buffetGain.connect(duck)
    startLoop(pink, windRumbleFilter, 0.0, true)

    // ---- Wind: stereo mid body pair with bank-reactive panning.
    const makeBody = (pan: number, offset: number): [BiquadFilterNode, GainNode, StereoPannerNode] => {
      const filter = ac.createBiquadFilter()
      filter.type = 'bandpass'
      filter.frequency.value = 480
      filter.Q.value = 0.8
      const gain = ac.createGain()
      gain.gain.value = 0
      const panner = ac.createStereoPanner()
      panner.pan.value = pan
      filter.connect(gain)
      gain.connect(panner)
      panner.connect(duck)
      startLoop(pink, filter, offset, true)
      return [filter, gain, panner]
    }
    const [windBodyFilterL, windBodyGainL, windBodyPanL] = makeBody(-0.35, 0.9)
    const [windBodyFilterR, windBodyGainR, windBodyPanR] = makeBody(0.35, 1.7)

    // ---- Wind: high-Q whistle that opens up in dives.
    const whistleFilter = ac.createBiquadFilter()
    whistleFilter.type = 'bandpass'
    whistleFilter.frequency.value = 2600
    whistleFilter.Q.value = 10
    const whistleGain = ac.createGain()
    whistleGain.gain.value = 0
    const whistlePan = ac.createStereoPanner()
    whistlePan.pan.value = 0.12
    whistleFilter.connect(whistleGain)
    whistleGain.connect(whistlePan)
    whistlePan.connect(duck)
    startLoop(white, whistleFilter, 0.4, true)

    // ---- Wind: turbulence band (banking / flap shed-air flutter).
    const turbFilter = ac.createBiquadFilter()
    turbFilter.type = 'bandpass'
    turbFilter.frequency.value = 900
    turbFilter.Q.value = 1.6
    const turbGain = ac.createGain()
    turbGain.gain.value = 0
    turbFilter.connect(turbGain)
    turbGain.connect(duck)
    startLoop(white, turbFilter, 1.3, true)

    // ---- Flap voice pool: always-running noise, gated by envelopes.
    // A shared low-pass gives every voice a low "air thump" branch.
    const flapThumpFilter = ac.createBiquadFilter()
    flapThumpFilter.type = 'lowpass'
    flapThumpFilter.frequency.value = 220
    flapThumpFilter.Q.value = 0.6
    const flapVoices: FlapVoice[] = []
    for (let i = 0; i < 4; i++) {
      const bp = ac.createBiquadFilter()
      bp.type = 'bandpass'
      bp.frequency.value = 500
      bp.Q.value = 1.2
      const bpGain = ac.createGain()
      bpGain.gain.value = 0
      bp.connect(bpGain)
      bpGain.connect(duck)
      startLoop(white, bp, 0.3 + i * 0.47)

      const lpGain = ac.createGain()
      lpGain.gain.value = 0
      flapThumpFilter.connect(lpGain)
      lpGain.connect(duck)
      flapVoices.push({ bp, bpGain, lpGain })
    }
    startLoop(pink, flapThumpFilter, 2.6)

    // ---- Surf: a deep low-passed bed plus a brighter "wash" band.
    const surfLowFilter = ac.createBiquadFilter()
    surfLowFilter.type = 'lowpass'
    surfLowFilter.frequency.value = 450
    surfLowFilter.Q.value = 0.4
    const surfLowGain = ac.createGain()
    surfLowGain.gain.value = 0
    surfLowFilter.connect(surfLowGain)
    surfLowGain.connect(duck)
    surfLowGain.connect(spaceSend)
    startLoop(pink, surfLowFilter, 2.2)

    const surfWashFilter = ac.createBiquadFilter()
    surfWashFilter.type = 'bandpass'
    surfWashFilter.frequency.value = 1100
    surfWashFilter.Q.value = 0.7
    const surfWashGain = ac.createGain()
    surfWashGain.gain.value = 0
    surfWashFilter.connect(surfWashGain)
    surfWashGain.connect(duck)
    startLoop(white, surfWashFilter, 0.8)

    // ---- Cicadas: two band-passed white banks, amplitude-modulated at two
    // different pulse rates so the shimmer reads as a chorus, not a buzzer.
    const cicadaGain = ac.createGain()
    cicadaGain.gain.value = 0
    cicadaGain.connect(duck)
    const makeCicadaBank = (
      freq: number,
      q: number,
      amRate: number,
      offset: number,
    ): BiquadFilterNode => {
      const filter = ac.createBiquadFilter()
      filter.type = 'bandpass'
      filter.frequency.value = freq
      filter.Q.value = q
      const am = ac.createGain()
      am.gain.value = 0.55
      const lfo = ac.createOscillator()
      lfo.type = 'sine'
      lfo.frequency.value = amRate
      const lfoGain = ac.createGain()
      lfoGain.gain.value = 0.45
      lfo.connect(lfoGain)
      lfoGain.connect(am.gain)
      lfo.start()
      filter.connect(am)
      am.connect(cicadaGain)
      startLoop(white, filter, offset)
      return filter
    }
    const cicadaFilterA = makeCicadaBank(6100, 2.6, 61, 1.1)
    makeCicadaBank(4300, 3.2, 47, 1.9)

    // ---- Kookaburra: two detuned sawtooths through two parallel formant
    // band-passes (swept during the laugh), then a distance low-pass. Idles
    // at gain 0; the wet send puts the laugh out in the landscape.
    const kookaOscA = ac.createOscillator()
    kookaOscA.type = 'sawtooth'
    kookaOscA.frequency.value = 600
    const kookaOscB = ac.createOscillator()
    kookaOscB.type = 'sawtooth'
    kookaOscB.frequency.value = 604
    kookaOscB.detune.value = 9
    const kookaFormant1 = ac.createBiquadFilter()
    kookaFormant1.type = 'bandpass'
    kookaFormant1.frequency.value = 1100
    kookaFormant1.Q.value = 3.5
    const kookaFormant2 = ac.createBiquadFilter()
    kookaFormant2.type = 'bandpass'
    kookaFormant2.frequency.value = 2400
    kookaFormant2.Q.value = 5.5
    const kookaDistance = ac.createBiquadFilter()
    kookaDistance.type = 'lowpass'
    kookaDistance.frequency.value = 2500
    kookaDistance.Q.value = 0.3
    const kookaGain = ac.createGain()
    kookaGain.gain.value = 0
    const kookaPan = ac.createStereoPanner()
    kookaOscA.connect(kookaFormant1)
    kookaOscA.connect(kookaFormant2)
    kookaOscB.connect(kookaFormant1)
    kookaOscB.connect(kookaFormant2)
    kookaFormant1.connect(kookaDistance)
    kookaFormant2.connect(kookaDistance)
    kookaDistance.connect(kookaGain)
    kookaGain.connect(kookaPan)
    kookaPan.connect(duck)
    kookaPan.connect(spaceSend)
    kookaOscA.start()
    kookaOscB.start()

    // ---- Generic distant chirp: a soft sine whistle, far away, mostly wet.
    const chirpOsc = ac.createOscillator()
    chirpOsc.type = 'sine'
    chirpOsc.frequency.value = 2400
    const chirpDistance = ac.createBiquadFilter()
    chirpDistance.type = 'lowpass'
    chirpDistance.frequency.value = 4200
    chirpDistance.Q.value = 0.2
    const chirpGain = ac.createGain()
    chirpGain.gain.value = 0
    const chirpPan = ac.createStereoPanner()
    chirpOsc.connect(chirpDistance)
    chirpDistance.connect(chirpGain)
    chirpGain.connect(chirpPan)
    chirpPan.connect(duck)
    chirpPan.connect(spaceSend)
    chirpOsc.start()

    return {
      duck,
      spaceSend,
      windRumbleFilter,
      windRumbleGain,
      buffetGain,
      windBodyFilterL,
      windBodyGainL,
      windBodyPanL,
      windBodyFilterR,
      windBodyGainR,
      windBodyPanR,
      whistleFilter,
      whistleGain,
      turbFilter,
      turbGain,
      windSources,
      flapVoices,
      surfLowGain,
      surfLowFilter,
      surfWashGain,
      cicadaGain,
      cicadaFilterA,
      kookaOscA,
      kookaOscB,
      kookaFormant1,
      kookaFormant2,
      kookaGain,
      kookaPan,
      chirpOsc,
      chirpGain,
      chirpPan,
    }
  }

  // -------------------------------------------------------------------------
  // Procedural buffers
  // -------------------------------------------------------------------------

  /** 1/f "pink" noise buffer via the Paul Kellet filter approximation. */
  private makePinkNoiseBuffer(ac: AudioContext, seconds: number): AudioBuffer {
    const length = Math.floor(ac.sampleRate * seconds)
    const buffer = ac.createBuffer(1, length, ac.sampleRate)
    const data = buffer.getChannelData(0)
    let b0 = 0
    let b1 = 0
    let b2 = 0
    let b3 = 0
    let b4 = 0
    let b5 = 0
    let b6 = 0
    for (let i = 0; i < length; i++) {
      const w = Math.random() * 2 - 1
      b0 = 0.99886 * b0 + w * 0.0555179
      b1 = 0.99332 * b1 + w * 0.0750759
      b2 = 0.969 * b2 + w * 0.153852
      b3 = 0.8665 * b3 + w * 0.3104856
      b4 = 0.55 * b4 + w * 0.5329522
      b5 = -0.7616 * b5 - w * 0.016898
      data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
      b6 = w * 0.115926
    }
    return buffer
  }

  /** Flat white noise buffer. */
  private makeWhiteNoiseBuffer(ac: AudioContext, seconds: number): AudioBuffer {
    const length = Math.floor(ac.sampleRate * seconds)
    const buffer = ac.createBuffer(1, length, ac.sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < length; i++) data[i] = Math.random() * 2 - 1
    return buffer
  }

  /**
   * Stereo "outdoor space" impulse response: exponentially decaying,
   * one-pole low-passed noise. Reads as open air with distant reflections,
   * not a room.
   */
  private makeSpaceImpulse(ac: AudioContext, seconds: number): AudioBuffer {
    const length = Math.floor(ac.sampleRate * seconds)
    const buffer = ac.createBuffer(2, length, ac.sampleRate)
    for (let ch = 0; ch < 2; ch++) {
      const data = buffer.getChannelData(ch)
      let lp = 0
      for (let i = 0; i < length; i++) {
        const decay = Math.pow(1 - i / length, 2.4)
        const w = (Math.random() * 2 - 1) * decay
        lp += 0.18 * (w - lp) // darken the tail like air absorption
        data[i] = lp * 0.9
      }
    }
    return buffer
  }

  // -------------------------------------------------------------------------
  // Wing flaps
  // -------------------------------------------------------------------------

  /**
   * Watches flapPhase for the downstroke crossing and fires a whoosh voice.
   * Uses an unwrapped phase accumulator so the 1→0 wrap never drops a beat.
   */
  private detectFlap(ac: AudioContext, g: AudioGraph, ctx: GameContext): void {
    const phase = ctx.flight.flapPhase
    let delta = phase - this.prevFlapPhase
    this.prevFlapPhase = phase
    if (delta < -0.5) delta += 1 // wrapped past 1
    if (delta <= 0) return // gliding, or jitter

    this.flapAcc += delta
    const index = Math.floor(this.flapAcc - AudioSystem.DOWNSTROKE)
    if (index <= this.lastFlapIndex) return
    this.lastFlapIndex = index

    const strength = ctx.flight.flapStrength
    if (strength < 0.05 || !ctx.flying) return
    this.triggerFlap(ac, g, ctx.species, strength)
  }

  /**
   * Envelopes one pooled voice open: a downward-swept band-pass feather
   * whoosh over a short low air thump, both scaled by flap strength.
   */
  private triggerFlap(ac: AudioContext, g: AudioGraph, species: SpeciesId, strength: number): void {
    const p = FLAP_PROFILES[species]
    const voice = g.flapVoices[this.nextVoice]
    this.nextVoice = (this.nextVoice + 1) % g.flapVoices.length

    const now = ac.currentTime
    const peak = p.gain * (0.35 + 0.65 * strength)
    const dur = p.dur * (0.85 + 0.3 * (1 - strength))

    voice.bp.Q.value = p.q
    voice.bp.frequency.cancelScheduledValues(now)
    voice.bp.frequency.setValueAtTime(p.freq * 1.6, now)
    voice.bp.frequency.exponentialRampToValueAtTime(p.freq * 0.65, now + dur)

    const whoosh = voice.bpGain.gain
    whoosh.cancelScheduledValues(now)
    whoosh.setValueAtTime(0.0001, now)
    whoosh.exponentialRampToValueAtTime(Math.max(peak, 0.001), now + dur * 0.25)
    whoosh.exponentialRampToValueAtTime(0.0001, now + dur)

    const thump = voice.lpGain.gain
    const thumpPeak = Math.max(peak * p.thump, 0.001)
    thump.cancelScheduledValues(now)
    thump.setValueAtTime(0.0001, now)
    thump.exponentialRampToValueAtTime(thumpPeak, now + 0.02)
    thump.exponentialRampToValueAtTime(0.0001, now + Math.max(dur * 0.6, 0.08))
  }

  // -------------------------------------------------------------------------
  // Distant calls
  // -------------------------------------------------------------------------

  /**
   * A soft distant call from a random far-off direction. Alternates between
   * a rising two-note whistle and a quick three-note descending warble.
   */
  private triggerChirp(ac: AudioContext, g: AudioGraph): void {
    const now = ac.currentTime
    const pan = (Math.random() * 2 - 1) * 0.85
    const base = 1800 + Math.random() * 1400
    const level = 0.016 + Math.random() * 0.018

    g.chirpPan.pan.setValueAtTime(pan, now)
    const freq = g.chirpOsc.frequency
    const gain = g.chirpGain.gain
    freq.cancelScheduledValues(now)
    gain.cancelScheduledValues(now)
    gain.setValueAtTime(0.0001, now)

    if (Math.random() < 0.55) {
      // Rising two-note whistle.
      freq.setValueAtTime(base, now)
      freq.exponentialRampToValueAtTime(base * 1.35, now + 0.09)
      gain.exponentialRampToValueAtTime(level, now + 0.03)
      gain.exponentialRampToValueAtTime(0.0001, now + 0.13)
      const t2 = now + 0.22
      freq.setValueAtTime(base * 1.2, t2)
      freq.exponentialRampToValueAtTime(base * 0.8, t2 + 0.14)
      gain.setValueAtTime(0.0001, t2)
      gain.exponentialRampToValueAtTime(level * 0.8, t2 + 0.03)
      gain.exponentialRampToValueAtTime(0.0001, t2 + 0.18)
    } else {
      // Quick descending three-note warble.
      let t = now
      for (let i = 0; i < 3; i++) {
        const f = base * (1.25 - 0.18 * i)
        freq.setValueAtTime(f * 1.1, t)
        freq.exponentialRampToValueAtTime(f * 0.92, t + 0.07)
        gain.setValueAtTime(0.0001, t)
        gain.exponentialRampToValueAtTime(level * (1 - 0.2 * i), t + 0.02)
        gain.exponentialRampToValueAtTime(0.0001, t + 0.09)
        t += 0.12
      }
    }
  }

  /**
   * A kookaburra laugh approximation: an accelerating, rising
   * "kook-kook-ka-ka-ka" run of sawtooth notes through two formant
   * band-passes whose center frequencies sweep open across the laugh, then
   * a slower falling chuckle tail. Quiet, low-passed, panned, and washed
   * through the space convolver so it reads as far away.
   */
  private triggerKookaburra(ac: AudioContext, g: AudioGraph): void {
    const now = ac.currentTime
    g.kookaPan.pan.setValueAtTime((Math.random() * 2 - 1) * 0.8, now)

    const freqA = g.kookaOscA.frequency
    const freqB = g.kookaOscB.frequency
    const gain = g.kookaGain.gain
    const f1 = g.kookaFormant1.frequency
    const f2 = g.kookaFormant2.frequency
    freqA.cancelScheduledValues(now)
    freqB.cancelScheduledValues(now)
    gain.cancelScheduledValues(now)
    f1.cancelScheduledValues(now)
    f2.cancelScheduledValues(now)
    gain.setValueAtTime(0.0001, now)

    const level = 0.026
    let t = now + 0.05

    // Formant sweep: the "mouth" opens as the laugh climbs, closes on the tail.
    f1.setValueAtTime(950, t)
    f2.setValueAtTime(2100, t)

    // Phase 1: accelerando, pitch climbing — the classic rising laugh.
    const riseNotes = 9
    for (let i = 0; i < riseNotes; i++) {
      const k = i / (riseNotes - 1)
      const noteDur = 0.26 - 0.13 * k
      const f = 470 + 430 * k
      const peak = level * (0.4 + 0.6 * k)
      freqA.setValueAtTime(f * 1.28, t)
      freqA.exponentialRampToValueAtTime(f * 0.84, t + noteDur * 0.85)
      freqB.setValueAtTime(f * 1.29, t)
      freqB.exponentialRampToValueAtTime(f * 0.85, t + noteDur * 0.85)
      f1.linearRampToValueAtTime(950 + 450 * k, t + noteDur)
      f2.linearRampToValueAtTime(2100 + 800 * k, t + noteDur)
      gain.setValueAtTime(0.0001, t)
      gain.exponentialRampToValueAtTime(peak, t + 0.022)
      gain.exponentialRampToValueAtTime(0.0001, t + noteDur * 0.9)
      t += noteDur
    }

    // Phase 2: the falling chuckle tail, slower and softer.
    const tailNotes = 4
    for (let i = 0; i < tailNotes; i++) {
      const k = i / (tailNotes - 1)
      const noteDur = 0.2 + 0.1 * k
      const f = 760 - 300 * k
      const peak = level * (0.7 - 0.5 * k)
      freqA.setValueAtTime(f * 1.2, t)
      freqA.exponentialRampToValueAtTime(f * 0.8, t + noteDur * 0.85)
      freqB.setValueAtTime(f * 1.21, t)
      freqB.exponentialRampToValueAtTime(f * 0.81, t + noteDur * 0.85)
      f1.linearRampToValueAtTime(1400 - 500 * k, t + noteDur)
      f2.linearRampToValueAtTime(2900 - 900 * k, t + noteDur)
      gain.setValueAtTime(0.0001, t)
      gain.exponentialRampToValueAtTime(Math.max(peak, 0.001), t + 0.03)
      gain.exponentialRampToValueAtTime(0.0001, t + noteDur * 0.9)
      t += noteDur + 0.04
    }
  }
}
