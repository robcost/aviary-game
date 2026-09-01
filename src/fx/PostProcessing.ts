/**
 * Cinematic post-processing pipeline for Aviary.
 *
 * The chain renders the scene into HDR (half-float) buffers and composes,
 * in order:
 *
 * 1. `RenderPass` — forward render of the scene into the composer's HDR buffer.
 * 2. `N8AOPostPass` — screen-space ambient occlusion tuned for aerial views.
 * 3. Depth of field (photo mode only) — focused on the terrain under the
 *    frame center, or on the bird when it is the centered subject.
 * 4. God rays + selective high-threshold bloom + AgX tone map + color grade.
 * 5. SMAA on the tone-mapped, display-referred image.
 * 6. Chromatic aberration (scales with airspeed) + vignette + film grain.
 *
 * Color-space policy: the renderer's own tone mapping is disabled at init so
 * every pass before step 4 works in scene-referred linear HDR. The
 * `ToneMappingEffect` (AgX) maps to display range in-chain, the grade runs on
 * the tone-mapped result, and postprocessing encodes to sRGB in the final
 * pass (the renderer keeps `outputColorSpace = SRGBColorSpace`).
 *
 * AgX over ACES: ACES clips bright warm sky to flat cream well before its
 * white point, which blew out the sunward hemisphere in review stills. AgX
 * compresses the same radiance over a much longer shoulder with per-channel
 * desaturation toward white, so the sky gradient survives all the way into
 * the sun glare and nothing in the frame reaches a clipped constant.
 *
 * Resolution policy: the composer's buffers are sized from
 * `renderer.getDrawingBufferSize` (postprocessing does this both at
 * construction and in `setSize`), so the whole chain runs at the full
 * device-pixel resolution set in main.ts (`setPixelRatio(min(dpr, 2))`).
 * Only AO, god rays, and DoF bokeh internally run half-res by design.
 *
 * This module is the ONLY one allowed to call `renderer.render` / compose the
 * frame.
 */
import * as THREE from 'three'
import {
  BlendFunction,
  BloomEffect,
  ChromaticAberrationEffect,
  DepthOfFieldEffect,
  Effect,
  EffectComposer,
  EffectPass,
  GodRaysEffect,
  KernelSize,
  type Pass,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
} from 'postprocessing'
// @ts-expect-error -- n8ao ships no type declarations; narrowed via AmbientOcclusionPass below.
import { N8AOPostPass } from 'n8ao'
import { WORLD, type GameContext, type GameModule } from '../core/GameState'

/**
 * Structural type for the members of n8ao's `N8AOPostPass` this module uses.
 * The package ships no declarations, so its constructor is cast to this shape.
 */
interface AmbientOcclusionPass extends Pass {
  /** When true, n8ao rescans the scene every frame for transparent materials. */
  autoDetectTransparency: boolean
  /** Live tuning proxy; property assignments reconfigure the pass. */
  configuration: {
    aoSamples: number
    aoRadius: number
    denoiseSamples: number
    denoiseRadius: number
    distanceFalloff: number
    intensity: number
    color: THREE.Color
    screenSpaceRadius: boolean
    halfRes: boolean
    depthAwareUpsampling: boolean
    transparencyAware: boolean
  }
}

/** Typed handle on the untyped n8ao constructor. */
const AmbientOcclusionPassCtor = N8AOPostPass as new (
  scene: THREE.Scene,
  camera: THREE.Camera,
  width: number,
  height: number,
) => AmbientOcclusionPass

/** Distance from the camera to the god-rays sun impostor, meters. */
const SUN_DISTANCE = 24000

/**
 * Radius of the sun impostor disc, meters (~0.26 degrees of half-arc — the
 * real sun's apparent size). A physically small disc keeps the pre-tonemap
 * HDR energy localized so bloom and god rays glow instead of whiting out.
 */
const SUN_RADIUS = 110

/**
 * Clamped HDR radiance of the sun disc (26:21:14 warm ratio, scaled down).
 * ~4x the bloom threshold: bright enough to bloom and drive god rays, small
 * enough that the AgX shoulder rolls it off instead of saturating.
 */
const SUN_RADIANCE = { r: 6.0, g: 4.85, b: 3.25 } as const

/**
 * Display exposure for the in-chain AgX tone map. AgX applies exposure as a
 * plain linear multiply (no ACES-style `/ 0.6` hidden gain), so 0.68 is a
 * true ~-1 EV cut from the previous effective 1.2x ACES gain that clipped
 * the sunward half of review frames. Combined with AgX's long desaturating
 * shoulder, sky radiance now grades smoothly into the sun instead of
 * slamming into flat cream.
 */
const TONEMAP_EXPOSURE = 0.68

/** Upper bound on god-rays screen-blend opacity, even staring into the sun. */
const GOD_RAYS_MAX_OPACITY = 0.38

/** Photo-mode focus raymarch: range cap and adaptive step tuning, meters. */
const FOCUS_MAX_DISTANCE = 6000
const FOCUS_STEP_MIN = 2
const FOCUS_STEP_GROWTH = 0.05

/**
 * Photographic depth-of-field model for photo mode. `focusRange` (the
 * distance over which CoC ramps to full bokeh) is `focus^2 / HYPERFOCAL`,
 * clamped below by {@link FOCUS_RANGE_MIN}:
 * - Close-up (focus 10 m): range 3 m — creamy subject-isolating bokeh.
 * - Midground (focus 100 m): range ~167 m — gentle near/far falloff.
 * - Landscape (focus 1 km+): range 16 km+ — the whole frame is sharp,
 *   exactly like a real lens stopped down past its hyperfocal distance.
 * The old `range = focus` linear ramp blurred everything off the exact
 * focal plane in every shot; nothing in the frame ever read as sharp.
 */
const FOCUS_HYPERFOCAL = 60
const FOCUS_RANGE_MIN = 3

/** The bird takes focus only when farther than this from the camera, meters. */
const BIRD_FOCUS_MIN_DISTANCE = 4

/**
 * Cosine of the max angle (18 deg) off frame center for bird auto-focus.
 * Wide enough to cover rule-of-thirds compositions — the old 6 deg cone
 * missed the bird in off-center photo presets and dropped focus onto the
 * terrain kilometers behind it, blurring the subject.
 */
const BIRD_FOCUS_COS = Math.cos(THREE.MathUtils.degToRad(18))

/** Airspeed (m/s) at which chromatic aberration reaches full strength. */
const ABERRATION_FULL_SPEED = 45

/** Chromatic aberration offset at rest / at {@link ABERRATION_FULL_SPEED}. */
const ABERRATION_MIN = 0.00035
const ABERRATION_MAX = 0.0015

/**
 * LDR color grade applied after tone mapping: lifted warm shadows, golden
 * midtones, a gentle teal pull in cool regions (sky and water reflections),
 * a soft filmic S-curve, and a small saturation push. Pure constants — the
 * grade is a fixed LUT-free tone curve, so it costs no uniform updates.
 */
const COLOR_GRADE_FRAGMENT = /* glsl */ `
  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    vec3 c = inputColor.rgb;
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

    // Lift the shadows toward warm gold (late-afternoon bounce light).
    float luma = dot(c, LUMA);
    float shadowMask = 1.0 - smoothstep(0.0, 0.42, luma);
    c += vec3(0.030, 0.019, 0.005) * shadowMask;

    // Golden Australian midtone warmth.
    c *= mix(vec3(1.0), vec3(1.045, 1.008, 0.940), 0.85);

    // Gentle teal in cool (blue-dominant) regions: sky and its reflections.
    float coolness = clamp((c.b - c.r) * 2.5, 0.0, 1.0);
    c = mix(c, c * vec3(0.945, 1.015, 1.030), coolness * 0.4);

    // Soft filmic S-curve for contrast, then a small saturation push.
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.20);
    float luma2 = dot(c, LUMA);
    c = mix(vec3(luma2), c, 1.07);

    outputColor = vec4(clamp(c, 0.0, 1.0), inputColor.a);
  }
`

/** Warm golden-hour color grade, implemented as a LUT-free tone curve. */
class ColorGradeEffect extends Effect {
  /** Builds the grade. It fully replaces the input color (SRC blend). */
  constructor() {
    super('ColorGradeEffect', COLOR_GRADE_FRAGMENT, { blendFunction: BlendFunction.SRC })
  }
}

/**
 * Animated hash-noise film grain. `time` is a built-in uniform provided by
 * postprocessing's effect shader header.
 */
const NOISE_FRAGMENT = /* glsl */ `
  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    float n = fract(sin(dot(uv + fract(time), vec2(12.9898, 78.233))) * 43758.5453);
    outputColor = vec4(vec3(n) * inputColor.rgb, inputColor.a);
  }
`

/**
 * Very subtle film grain. Premultiplying the noise by the input color and
 * screen-blending it reads as camera grain in the mids without lifting the
 * black level.
 */
class FilmGrainEffect extends Effect {
  /** Builds the grain with a fixed, low opacity. */
  constructor() {
    super('FilmGrainEffect', NOISE_FRAGMENT, { blendFunction: BlendFunction.SCREEN })
    this.blendMode.opacity.value = 0.09
  }
}

/**
 * The frame composer. Owns the `EffectComposer` chain and is the single
 * module that renders. `update` retunes the speed- and sun-dependent effects
 * each frame with zero allocations; `render` composes the final image.
 */
export class PostProcessing implements GameModule {
  readonly name = 'post'

  private ctx!: GameContext
  private composer!: EffectComposer

  /** Bright HDR impostor the god-rays effect traces from. Follows the sun. */
  private sunMesh!: THREE.Mesh
  private godRays!: GodRaysEffect
  private chromaticAberration!: ChromaticAberrationEffect
  private dof!: DepthOfFieldEffect
  private dofPass!: EffectPass

  /** Scratch vectors — reused every frame, zero per-frame allocations. */
  private readonly scratchDir = new THREE.Vector3()
  private readonly scratchPos = new THREE.Vector3()
  private readonly scratchBird = new THREE.Vector3()

  /** Build the whole chain. Called once, after the world modules have init'd. */
  init(ctx: GameContext): void {
    this.ctx = ctx
    const { renderer, scene, camera } = ctx

    // Tone map in-chain instead: scene passes must stay scene-referred linear.
    // The exposure still comes from the renderer — three uploads
    // `toneMappingExposure` to every program, including the effect material
    // that hosts the in-chain AgX map (see {@link TONEMAP_EXPOSURE}).
    renderer.toneMapping = THREE.NoToneMapping
    renderer.toneMappingExposure = TONEMAP_EXPOSURE

    this.sunMesh = this.createSunImpostor()
    scene.add(this.sunMesh)

    this.composer = new EffectComposer(renderer, {
      frameBufferType: THREE.HalfFloatType,
      stencilBuffer: false,
      multisampling: 0,
    })

    this.composer.addPass(new RenderPass(scene, camera))
    this.composer.addPass(this.createAmbientOcclusionPass())

    // Photo-mode depth of field. DoF is a convolution effect, so it needs its
    // own (usually disabled) pass; it runs before tone mapping so the bokeh
    // integrates HDR energy correctly. Focus is driven manually each photo
    // frame (see updateFocus): a terrain raymarch under the frame center,
    // overridden by the bird when it is a near-center subject. The library's
    // `target` auto-focus is deliberately unused — in a first-person sim the
    // camera rides the bird, so targeting it put the focal plane at ~0 m and
    // blurred the entire frame uniformly. CoC focus/range values are
    // world-space meters (verified against the 6.37 CoC shader: it compares
    // view-space distance directly). The range follows the hyperfocal model
    // in {@link FOCUS_HYPERFOCAL}, so landscape shots are sharp edge to edge
    // and only close subjects get bokeh. Max CoC is capped at ~5 px at 1080p
    // (bokehScale 2.6 at half res) so even fully defocused areas stay
    // readable.
    this.dof = new DepthOfFieldEffect(camera, {
      focusDistance: 60,
      focusRange: 60,
      bokehScale: 2.6,
      resolutionScale: 0.5,
    })
    this.dofPass = new EffectPass(camera, this.dof)
    this.dofPass.enabled = false
    this.composer.addPass(this.dofPass)

    // HDR light transport (god rays, selective bloom), then tone map + grade.
    // God rays stay a subtle atmospheric accent: low weight and a hard
    // clampMax keep the screen-blended ray texture from whiting out the
    // sunward half of the frame before the tone map can roll it off.
    this.godRays = new GodRaysEffect(camera, this.sunMesh, {
      samples: 48,
      density: 0.93,
      decay: 0.92,
      weight: 0.2,
      exposure: 0.3,
      clampMax: 0.65,
      resolutionScale: 0.5,
      kernelSize: KernelSize.SMALL,
      blur: true,
    })
    const bloom = new BloomEffect({
      blendFunction: BlendFunction.ADD,
      mipmapBlur: true,
      // Above 1.2 only true HDR energy blooms: the clamped sun disc and
      // specular sun glints on water — never sky haze or diffuse terrain.
      // The wide smoothing is the soft knee: bloom fades in gradually above
      // the threshold instead of snapping whole sky regions into the glow.
      luminanceThreshold: 1.2,
      luminanceSmoothing: 0.55,
      intensity: 0.5,
      radius: 0.7,
    })
    const toneMapping = new ToneMappingEffect({ mode: ToneMappingMode.AGX })
    this.composer.addPass(new EffectPass(camera, this.godRays, bloom, toneMapping, new ColorGradeEffect()))

    // Anti-aliasing on the tone-mapped image. SMAA is a convolution effect
    // and must not share a pass with chromatic aberration (also convolution).
    this.composer.addPass(new EffectPass(camera, new SMAAEffect({ preset: SMAAPreset.HIGH })))

    // Finishing: speed-scaled chromatic aberration, vignette, film grain.
    this.chromaticAberration = new ChromaticAberrationEffect({
      offset: new THREE.Vector2(ABERRATION_MIN, ABERRATION_MIN),
      radialModulation: true,
      modulationOffset: 0.18,
    })
    const vignette = new VignetteEffect({ offset: 0.3, darkness: 0.5 })
    this.composer.addPass(new EffectPass(camera, this.chromaticAberration, vignette, new FilmGrainEffect()))

    addEventListener('resize', () => this.composer.setSize(innerWidth, innerHeight))
  }

  /** Retune the dynamic effects from the live flight and sun state. */
  update(_dt: number, ctx: GameContext): void {
    const { camera, sunDirection, flight } = ctx

    // Keep the sun impostor pinned to the sky, far along the sun direction.
    camera.getWorldPosition(this.scratchPos)
    this.sunMesh.position.copy(sunDirection).multiplyScalar(SUN_DISTANCE).add(this.scratchPos)

    // God rays only contribute while the sun is toward the view direction and
    // above the horizon; both terms fade smoothly so the effect never pops.
    camera.getWorldDirection(this.scratchDir)
    const facing = THREE.MathUtils.smoothstep(this.scratchDir.dot(sunDirection), -0.05, 0.3)
    const elevation = THREE.MathUtils.smoothstep(sunDirection.y, -0.05, 0.12)
    this.godRays.blendMode.opacity.value = GOD_RAYS_MAX_OPACITY * facing * elevation

    // Slight chromatic aberration that grows quadratically with airspeed.
    // Photo mode drops the speed term: the preset freezes physics with a
    // nonzero airspeed, and a still must not carry motion-speed fringing —
    // only the fixed lens-character minimum remains.
    const speedFactor = ctx.photoShot !== null ? 0 : Math.min(flight.speed / ABERRATION_FULL_SPEED, 1)
    const aberration = ABERRATION_MIN + (ABERRATION_MAX - ABERRATION_MIN) * speedFactor * speedFactor
    this.chromaticAberration.offset.set(aberration, aberration)

    // Depth of field only in photo mode, focused on the framed subject.
    this.dofPass.enabled = ctx.photoShot !== null
    if (this.dofPass.enabled) this.updateFocus(ctx)
  }

  /**
   * Drive the photo-mode focal plane. Focus lands on the terrain under the
   * frame center (adaptive raymarch against the exact terrain height field),
   * or on the bird when it is a distinct subject near the frame center. The
   * focus range follows the hyperfocal model ({@link FOCUS_HYPERFOCAL}):
   * `range = focus^2 / hyperfocal`, so close subjects get shallow bokeh and
   * landscape focus distances push the range past the whole frame — the shot
   * is sharp edge to edge, like a real lens stopped down. The CoC shader
   * ramps blur as `smoothstep(0, range, |distance - focus|)` in world-space
   * meters (verified against the 6.39 CircleOfConfusionMaterial shader).
   * Zero allocations: scratch vectors only.
   */
  private updateFocus(ctx: GameContext): void {
    // scratchPos / scratchDir already hold the camera position and view
    // direction — `update` fills them every frame before calling this.
    let focus = this.raymarchTerrain(ctx)

    // The bird wins focus when it is a real subject: near the frame center,
    // closer than the terrain hit, and not carrying the camera itself.
    this.scratchBird.copy(ctx.flight.position).sub(this.scratchPos)
    const birdDistance = this.scratchBird.length()
    if (birdDistance > BIRD_FOCUS_MIN_DISTANCE && birdDistance < focus) {
      const centered = this.scratchBird.divideScalar(birdDistance).dot(this.scratchDir)
      if (centered > BIRD_FOCUS_COS) focus = birdDistance
    }

    this.dof.cocMaterial.focusDistance = focus
    this.dof.cocMaterial.focusRange = Math.max(FOCUS_RANGE_MIN, (focus * focus) / FOCUS_HYPERFOCAL)
  }

  /**
   * Distance from the camera to the terrain (or ocean surface) under the
   * frame center, meters. Adaptive raymarch: the step grows with distance,
   * with a short bisection refine at the crossing. Capped at
   * {@link FOCUS_MAX_DISTANCE} for sky shots. ~150 height samples worst
   * case, photo mode only.
   */
  private raymarchTerrain(ctx: GameContext): number {
    let previous = FOCUS_STEP_MIN
    let t = FOCUS_STEP_MIN
    while (t < FOCUS_MAX_DISTANCE) {
      if (this.isBelowSurface(ctx, t)) {
        // Crossed the surface: bisect [previous, t] down to sub-meter focus.
        let lo = previous
        let hi = t
        for (let i = 0; i < 5; i++) {
          const mid = (lo + hi) * 0.5
          if (this.isBelowSurface(ctx, mid)) hi = mid
          else lo = mid
        }
        return (lo + hi) * 0.5
      }
      previous = t
      t += Math.max(FOCUS_STEP_MIN, t * FOCUS_STEP_GROWTH)
    }
    return FOCUS_MAX_DISTANCE
  }

  /**
   * True when the point `t` meters along the frame-center ray (scratchPos +
   * t * scratchDir) sits at or below the local surface. The terrain height is
   * clamped to sea level, so ocean shots focus on the water, not the seabed.
   */
  private isBelowSurface(ctx: GameContext, t: number): boolean {
    const o = this.scratchPos
    const d = this.scratchDir
    const surface = Math.max(ctx.getTerrainHeight(o.x + d.x * t, o.z + d.z * t), WORLD.seaLevel)
    return o.y + d.y * t <= surface
  }

  /** Compose and present the frame. Called once per frame, after updates. */
  render(dt: number): void {
    this.composer.render(dt)
  }

  /**
   * Create the emissive sun disc the god-rays effect traces from. Its HDR
   * radiance is clamped ({@link SUN_RADIANCE}) and its angular size matches
   * the real sun ({@link SUN_RADIUS}), so it feeds the high-threshold bloom
   * without saturating the sky around it; transparent with no depth write, as
   * `GodRaysEffect` requires of its light source. The sky's own sun glow
   * absorbs it visually, and bloom melts its silhouette into a glare.
   */
  private createSunImpostor(): THREE.Mesh {
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color().setRGB(SUN_RADIANCE.r, SUN_RADIANCE.g, SUN_RADIANCE.b),
      transparent: true,
      depthWrite: false,
      fog: false,
      toneMapped: false,
    })
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(SUN_RADIUS, 24, 16), material)
    mesh.frustumCulled = false
    return mesh
  }

  /**
   * Build the n8ao pass. Screen-space radius keeps occlusion visually
   * consistent from treetop height to soaring altitude across the 12 km
   * world; half-resolution AO with depth-aware upsampling stays inside the
   * 60 fps budget on Apple-silicon laptops.
   */
  private createAmbientOcclusionPass(): AmbientOcclusionPass {
    const pass = new AmbientOcclusionPassCtor(this.ctx.scene, this.ctx.camera, innerWidth, innerHeight)
    pass.autoDetectTransparency = false
    const cfg = pass.configuration
    cfg.transparencyAware = false
    cfg.screenSpaceRadius = true
    cfg.aoRadius = 40
    cfg.distanceFalloff = 0.2
    cfg.intensity = 2.2
    cfg.aoSamples = 12
    cfg.denoiseSamples = 8
    cfg.denoiseRadius = 12
    cfg.halfRes = true
    cfg.depthAwareUpsampling = true
    cfg.color.setRGB(0, 0, 0)
    return pass
  }
}
