/**
 * Birds module.
 *
 * Everything feathered in Aviary lives here:
 *
 * - **First-person rig** — the player IS the bird. A per-species procedural
 *   body kit is parented to the camera: a sculpted beak at the bottom of the
 *   frame, a soft domed breast shelf under it, scapular tufts at each wing
 *   root, and two fully articulated wings (shoulder → elbow → wrist chains
 *   with individually fanned primary feathers) sweeping through the
 *   peripheral view. Wing pose is driven by {@link FlightState.flapPhase} and
 *   {@link FlightState.flapStrength}: a fast, powerful downstroke, a relaxed
 *   folding upstroke, and a spread-fingered glide pose. The head
 *   counter-rolls slightly against banking, the way a real bird stabilises
 *   its gaze. The rig swaps instantly when {@link GameContext.species}
 *   changes.
 *
 * - **Ambient life** — a distant instanced flock of little corellas strung
 *   into a loose skein along a looping path over the river valley (per-bird
 *   path lag on the CPU, wing flapping in the vertex shader), and two
 *   wedge-tailed-eagle silhouettes soaring slow thermal circles over the
 *   north-west ranges.
 *
 * All plumage is procedural and gets a dedicated wing material stack, fully
 * distinct from terrain shading: painted feather albedo maps (rachis, angled
 * barb striations, staggered directional covert rows — no circular motifs)
 * with matching Sobel-derived normal maps, shared physical feather materials
 * tinted per species by vertex colour (low-gloss anisotropic specular and a
 * deliberately restrained sheen/rim response so grazing angles never wash the
 * plumage toward sky tones), a faint grazing-angle rim-light term that
 * separates the wing silhouette from the ground, and an attribute-gated
 * iridescent tint on green coverts (rainbow lorikeet). Colour zones blend
 * through smooth gradient ramps, root AO is baked into vertex colours so
 * wings seat into the body, and layered covert feather cards shingle over
 * the wing panels — with low-frequency inter-layer AO baked at every card
 * base and across the panel under the card stack — so the surface reads as
 * stacked feathers instead of a single textured plane. The wedge-tailed eagle carries an explicit pale
 * nape/covert band and pale covert-tip rows so its brown wings never merge
 * with dune terrain. No external assets. The per-frame path allocates
 * nothing — every update writes through module-level scratch objects.
 */
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { GameContext, GameModule, SpeciesId } from '../core/GameState'

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

const TWO_PI = Math.PI * 2

/** Fraction of the flap cycle spent on the (fast, powerful) downstroke. */
const DOWNSTROKE_FRACTION = 0.42

/** Phase lag of the elbow behind the shoulder, in cycle fractions. */
const ELBOW_LAG = 0.06

/** Phase lag of the wrist behind the shoulder, in cycle fractions. */
const WRIST_LAG = 0.13

/** Corella flock size. */
const FLOCK_COUNT = 48

/** Flock loop anchor, over the river valley north of spawn. */
const FLOCK_ANCHOR_X = 260
const FLOCK_ANCHOR_Z = -450

/** Flock loop ellipse radii, meters. */
const FLOCK_RADIUS_X = 850
const FLOCK_RADIUS_Z = 640

/** Flock height above terrain, meters. */
const FLOCK_CLEARANCE = 95

// ---------------------------------------------------------------------------
// Scratch objects — reused every frame so update() allocates nothing.
// ---------------------------------------------------------------------------

const _pos = new THREE.Vector3()
const _quat = new THREE.Quaternion()
const _euler = new THREE.Euler()
const _mat = new THREE.Matrix4()
const _unitScale = new THREE.Vector3(1, 1, 1)

// Init-time colour scratch (init may allocate; update may not).
const _cA = new THREE.Color()
const _cB = new THREE.Color()
const _cC = new THREE.Color()
const _cD = new THREE.Color()
const _cE = new THREE.Color()
const _cM = new THREE.Color()

// ---------------------------------------------------------------------------
// Species styling
// ---------------------------------------------------------------------------

/** Procedural beak parameters for one species. */
interface BeakStyle {
  /** Beak length, meters (first-person exaggerated scale). */
  length: number
  /** Width at the base, meters. */
  width: number
  /** Height at the base, meters. */
  height: number
  /** Downward curve of the tip (hook), meters. */
  hook: number
  /** Colour at the base (cere end). */
  baseColor: number
  /** Colour toward the tip. */
  tipColor: number
}

/** Full procedural styling for one playable species. */
interface SpeciesStyle {
  /** Half wingspan from the shoulder, meters. */
  halfSpan: number
  /** Number of fanned primary feathers per wing. */
  primaryCount: number
  /**
   * Wingtip shape: 0 = broad slotted "fingers" (eagle),
   * 1 = pointed swept tip (lorikeet).
   */
  pointedness: number
  /** Upper covert colour — the leading band the player sees most. */
  covert: number
  /** Mid-wing / secondary panel colour. */
  midWing: number
  /** Primary feather base colour. */
  primary: number
  /** Primary feather tip colour. */
  primaryTip: number
  /** Breast plumage colour. */
  breast: number
  /** Breast streak/barring colour. */
  breastStreak: number
  /** Covert-card tip colour — the visible feather-row banding on the wing. */
  covertTip: number
  /** Pale nape/covert band across the upper wing (wedge-tailed eagle). */
  napeBand?: { color: number; strength: number }
  /**
   * Deep colour patch at the wing root / leading edge — the mantle plumage
   * the bird sees where its own wing meets the body (rainbow lorikeet's
   * blue nape flowing onto the shoulder).
   */
  rootPatch?: { color: number; strength: number }
  /** Belly colour beyond the breast band (rainbow lorikeet's deep blue). */
  belly?: number
  /** Iridescent-sheen gate on the green coverts (rainbow lorikeet). */
  iridescence?: number
  beak: BeakStyle
}

/**
 * Colour and proportion table for the five playable species. Colours are the
 * *upper* wing surface — that is what the bird sees of itself in flight.
 */
const SPECIES_STYLE: Record<SpeciesId, SpeciesStyle> = {
  'wedge-tailed-eagle': {
    halfSpan: 1.05,
    primaryCount: 7,
    pointedness: 0.05,
    covert: 0x8a5a2f, // tawny nape-to-covert band
    midWing: 0x2e1e10, // dark chocolate
    primary: 0x1a1109,
    primaryTip: 0x0d0a06,
    breast: 0x30200e,
    breastStreak: 0x96682f,
    covertTip: 0xb08744, // pale feather-row tips banding the dark wing
    napeBand: { color: 0xc79a52, strength: 0.6 },
    beak: { length: 0.105, width: 0.026, height: 0.03, hook: 0.03, baseColor: 0x9a8f7c, tipColor: 0x2a241d },
  },
  'sulphur-crested-cockatoo': {
    halfSpan: 0.5,
    primaryCount: 6,
    pointedness: 0.35,
    covert: 0xf6f1e4,
    midWing: 0xece6d4,
    primary: 0xe9e0c2, // sulphur wash on the flight feathers
    primaryTip: 0xd8ce9e,
    breast: 0xf7f3e8,
    breastStreak: 0xe9e3d1,
    covertTip: 0xfdfaf0,
    beak: { length: 0.075, width: 0.024, height: 0.026, hook: 0.026, baseColor: 0x3c3c40, tipColor: 0x1f1f22 },
  },
  'rainbow-lorikeet': {
    halfSpan: 0.23,
    primaryCount: 5,
    pointedness: 0.8,
    covert: 0x2c8f2b, // bright green shoulder
    midWing: 0x1f7226,
    primary: 0x175c2c,
    primaryTip: 0x2c6f8a, // blue-washed tips
    breast: 0xe2641a, // orange breast fills the frame bottom
    breastStreak: 0xf0a01e,
    covertTip: 0x5fb63a, // yellow-green feather-row tips ramping the coverts
    iridescence: 0.9,
    beak: { length: 0.05, width: 0.014, height: 0.016, hook: 0.014, baseColor: 0xd5481f, tipColor: 0xb03214 },
  },
  galah: {
    halfSpan: 0.37,
    primaryCount: 6,
    pointedness: 0.55,
    covert: 0xbdbab2, // pale silver-grey coverts
    midWing: 0x908d87,
    primary: 0x767570,
    primaryTip: 0x5b5a56,
    breast: 0xd05f7f, // rose pink
    breastStreak: 0xe391a6,
    covertTip: 0xd6d3cb,
    beak: { length: 0.05, width: 0.016, height: 0.018, hook: 0.014, baseColor: 0xd9cfba, tipColor: 0xb5aa92 },
  },
  'laughing-kookaburra': {
    halfSpan: 0.32,
    primaryCount: 6,
    pointedness: 0.3,
    covert: 0x4aa3d6, // the famous blue wing flash
    midWing: 0x4a3a26,
    primary: 0x39301e,
    primaryTip: 0x261e12,
    breast: 0xe8dcc2, // cream
    breastStreak: 0xc4ae8a,
    covertTip: 0x8ed0f2, // pale-blue spangled covert tips over the wing flash
    beak: { length: 0.13, width: 0.036, height: 0.024, hook: 0.008, baseColor: 0x36311f, tipColor: 0x201b10 },
  },
}

// ---------------------------------------------------------------------------
// Small deterministic hash — cheap "noise" without allocations.
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random in [0, 1) from a seed number. */
function hash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453
  return s - Math.floor(s)
}

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Clamped smoothstep of t into 0..1. */
function smooth01(t: number): number {
  const c = Math.min(Math.max(t, 0), 1)
  return c * c * (3 - 2 * c)
}

// ---------------------------------------------------------------------------
// Canvas texture generation
// ---------------------------------------------------------------------------

/** Create a 2D canvas and its context, throwing if unavailable. */
function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const g = canvas.getContext('2d')
  if (!g) throw new Error('Birds: 2D canvas context unavailable')
  return [canvas, g]
}

/** An albedo texture paired with its Sobel-derived tangent-space normal map. */
interface FeatherMaps {
  map: THREE.CanvasTexture
  normalMap: THREE.CanvasTexture
}

/**
 * Derive a tangent-space normal map from a painted canvas via a Sobel filter
 * over its alpha-weighted luminance. The rachis ridge, barb striations, and
 * feather-scallop overlaps in the albedo become real surface relief so the
 * plumage catches light instead of reading as a flat print.
 */
function makeNormalMap(src: HTMLCanvasElement, strength: number, wrap: boolean): THREE.CanvasTexture {
  const w = src.width
  const h = src.height
  const sg = src.getContext('2d')
  if (!sg) throw new Error('Birds: 2D canvas context unavailable')
  const px = sg.getImageData(0, 0, w, h).data
  const height = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) {
    const a = px[i * 4 + 3] / 255
    height[i] = ((px[i * 4] * 0.299 + px[i * 4 + 1] * 0.587 + px[i * 4 + 2] * 0.114) / 255) * a
  }
  const at = (x: number, y: number): number => {
    if (wrap) {
      x = (x + w) % w
      y = (y + h) % h
    } else {
      x = Math.min(Math.max(x, 0), w - 1)
      y = Math.min(Math.max(y, 0), h - 1)
    }
    return height[y * w + x]
  }
  const [canvas, g] = makeCanvas(w, h)
  const img = g.createImageData(w, h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx =
        at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1))
      const dy =
        at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1))
      // Canvas rows run downward but the texture upload flips Y, so +dy in
      // canvas space is +v in texture space: keep green pointing up-vane.
      const nx = -dx * strength
      const ny = dy * strength
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1)
      const o = (y * w + x) * 4
      img.data[o] = Math.round((nx * inv * 0.5 + 0.5) * 255)
      img.data[o + 1] = Math.round((ny * inv * 0.5 + 0.5) * 255)
      img.data[o + 2] = Math.round((inv * 0.5 + 0.5) * 255)
      img.data[o + 3] = 255
    }
  }
  g.putImageData(img, 0, 0)
  const tex = new THREE.CanvasTexture(canvas)
  if (wrap) {
    tex.wrapS = THREE.RepeatWrapping
    tex.wrapT = THREE.RepeatWrapping
  }
  tex.anisotropy = 4
  tex.generateMipmaps = true
  tex.minFilter = THREE.LinearMipmapLinearFilter
  return tex
}

/**
 * A single-feather detail texture: an alpha-cut feather silhouette with a
 * central rachis and angled barb striations. Near-white so per-vertex species
 * colours tint it. Shared by every feather of every species, together with
 * its derived normal map.
 */
function makeFeatherTexture(): FeatherMaps {
  const W = 128
  const H = 256
  const [canvas, g] = makeCanvas(W, H)
  g.clearRect(0, 0, W, H)

  // Silhouette: base at the bottom of the canvas (v = 0), rounded tip at top.
  // Width profile — narrow quill, broad vane, rounded tip.
  const cx = W * 0.5
  const vaneHalfWidth = (t: number): number =>
    Math.max(W * 0.46 * Math.pow(Math.sin(Math.min(t * 1.12, 1) * Math.PI * 0.82), 0.65), W * 0.05 * (1 - t))
  const traceVane = (): void => {
    g.beginPath()
    g.moveTo(cx - W * 0.06, H)
    for (let i = 0; i <= 24; i++) {
      const t = i / 24
      g.lineTo(cx - vaneHalfWidth(t), H - t * H)
    }
    for (let i = 24; i >= 0; i--) {
      const t = i / 24
      g.lineTo(cx + vaneHalfWidth(t), H - t * H)
    }
    g.closePath()
  }
  traceVane()
  const grad = g.createLinearGradient(0, H, 0, 0)
  grad.addColorStop(0, '#e8e4db')
  grad.addColorStop(0.7, '#f1eee7')
  grad.addColorStop(1, '#ddd7cb')
  g.fillStyle = grad
  g.fill()

  // Clip further drawing to the vane.
  g.save()
  g.clip()

  // Barb striations, angled up-and-out from the rachis — dense dark grain
  // with sparser pale inter-barb lines so the vane reads as combed fibres.
  g.lineWidth = 0.9
  for (let i = 0; i < 150; i++) {
    const y = H - (i / 150) * H
    const a = 0.05 + hash(i * 3.7) * 0.07
    g.strokeStyle = `rgba(58, 50, 40, ${a.toFixed(3)})`
    g.beginPath()
    g.moveTo(cx, y)
    g.lineTo(0, y + H * 0.1)
    g.moveTo(cx, y)
    g.lineTo(W, y + H * 0.1)
    g.stroke()
  }
  g.lineWidth = 0.8
  for (let i = 0; i < 60; i++) {
    const y = H - ((i + 0.5) / 60) * H
    // Pale inter-barb lines stay dim and warm — bright lines here read as
    // white piping on the finished feather.
    const a = 0.02 + hash(i * 8.1) * 0.03
    g.strokeStyle = `rgba(232, 222, 206, ${a.toFixed(3)})`
    g.beginPath()
    g.moveTo(cx, y)
    g.lineTo(0, y + H * 0.1)
    g.moveTo(cx, y)
    g.lineTo(W, y + H * 0.1)
    g.stroke()
  }

  // Edge shading: a soft dark inner rim along the silhouette. The alpha-cut
  // edge grades darker, never brighter — no white piping on the cut.
  traceVane()
  g.strokeStyle = 'rgba(48, 40, 32, 0.3)'
  g.lineWidth = 6
  g.stroke()

  // Rachis — a tapering warm shaft with dark groove edges, kept well below
  // white so the specular highlight, not the albedo, picks it out.
  const shaft = g.createLinearGradient(cx - 3, 0, cx + 3, 0)
  shaft.addColorStop(0, 'rgba(74, 63, 49, 0.4)')
  shaft.addColorStop(0.5, 'rgba(238, 230, 216, 0.6)')
  shaft.addColorStop(1, 'rgba(74, 63, 49, 0.4)')
  g.fillStyle = shaft
  g.beginPath()
  g.moveTo(cx - 2.8, H)
  g.lineTo(cx - 0.8, 0)
  g.lineTo(cx + 0.8, 0)
  g.lineTo(cx + 2.8, H)
  g.closePath()
  g.fill()
  g.restore()

  // Vane splits — real alpha cuts through the vane so every feather card
  // breaks the silhouette with ragged trailing-edge notches.
  g.globalCompositeOperation = 'destination-out'
  g.strokeStyle = 'rgba(0, 0, 0, 1)'
  for (let i = 0; i < 8; i++) {
    const y = H * (0.06 + hash(i * 11.3) * 0.5)
    const side = hash(i * 5.9) > 0.5 ? 1 : -1
    const w = vaneHalfWidth(1 - y / H)
    g.lineWidth = 1.3 + hash(i * 2.3) * 1.2
    g.beginPath()
    g.moveTo(cx + side * w * 0.4, y)
    g.lineTo(cx + side * (w + 4), y + H * 0.045)
    g.stroke()
  }
  g.globalCompositeOperation = 'source-over'

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  tex.generateMipmaps = true
  tex.minFilter = THREE.LinearMipmapLinearFilter
  return { map: tex, normalMap: makeNormalMap(canvas, 2.2, false) }
}

/**
 * A tileable plumage micro-detail texture built the way real coverts read:
 * staggered feather rows flowing down the texture, each feather a soft
 * directional tongue with a faint rachis line and angled barb striations, and
 * a low-frequency shadow pocket where each row slips under the row above.
 * There are no closed circles anywhere — every mark is directional, so the
 * surface reads as combed plumage, never as scale mail or knotwork.
 * Near-white overall so per-vertex species colours dominate. Used by the wing
 * panels, the scapular tufts, and the first-person breast, together with its
 * derived normal map.
 */
function makePlumageTexture(): FeatherMaps {
  const S = 256
  const [canvas, g] = makeCanvas(S, S)
  g.fillStyle = '#f2efe9'
  g.fillRect(0, 0, S, S)

  // Staggered feather rows. Rows run across (x), plumage flow runs down (y).
  // Iterating one row/col beyond each edge keeps the pattern seamless under
  // RepeatWrapping.
  const rows = 8
  const cols = 8
  const fw = S / cols // feather width
  const fh = S / rows // row pitch
  for (let r = -1; r <= rows; r++) {
    for (let c = -1; c <= cols; c++) {
      const jitter = hash(r * 17.3 + c * 7.7)
      const cx = (c + (r % 2 ? 0.5 : 0)) * fw + (jitter - 0.5) * 4
      const y0 = r * fh + (jitter - 0.5) * 3 // feather root (tucks under row above)
      const len = fh * (1.55 + jitter * 0.35) // overlaps the next row down

      // Low-frequency inter-row AO: a soft horizontal shadow pocket at the
      // root, where this feather slides under the row above.
      const pocket = g.createLinearGradient(0, y0 - fh * 0.12, 0, y0 + fh * 0.5)
      pocket.addColorStop(0, 'rgba(64, 54, 42, 0.15)')
      pocket.addColorStop(1, 'rgba(64, 54, 42, 0)')
      g.fillStyle = pocket
      g.fillRect(cx - fw * 0.62, y0 - fh * 0.12, fw * 1.24, fh * 0.62)

      // Rachis: one faint warm line down the feather centre.
      g.strokeStyle = 'rgba(88, 74, 58, 0.14)'
      g.lineWidth = 1.1
      g.beginPath()
      g.moveTo(cx, y0 + fh * 0.1)
      g.lineTo(cx + (jitter - 0.5) * 3, y0 + len)
      g.stroke()

      // Barb striations: short strokes angling down-and-out from the rachis,
      // denser toward the tip. Alternating faint dark/pale keeps the grain
      // visible without ever approaching white.
      const barbs = 7
      for (let b = 0; b < barbs; b++) {
        const t = (b + 0.5) / barbs
        const by = y0 + fh * 0.15 + t * (len - fh * 0.2)
        const reach = fw * (0.34 + 0.2 * Math.sin(t * Math.PI)) * (0.85 + jitter * 0.3)
        const droop = fh * 0.32
        for (const side of [-1, 1]) {
          const aDark = 0.05 + hash(r * 3.1 + c * 5.7 + b * 1.9 + side) * 0.06
          g.strokeStyle = `rgba(70, 60, 47, ${aDark.toFixed(3)})`
          g.lineWidth = 0.9
          g.beginPath()
          g.moveTo(cx, by)
          g.lineTo(cx + side * reach, by + droop)
          g.stroke()
        }
      }

      // Feather tip: a soft dark under-edge where this feather's rounded end
      // lies over the row below — an open downward tongue, not a circle. Two
      // slightly offset strokes give it thickness without hard lines.
      const tipY = y0 + len
      for (let k = 0; k < 2; k++) {
        g.strokeStyle = `rgba(58, 48, 38, ${(0.11 - k * 0.045).toFixed(3)})`
        g.lineWidth = 2.2 + k * 2
        g.beginPath()
        g.moveTo(cx - fw * 0.46, tipY - fh * 0.42)
        g.quadraticCurveTo(cx, tipY + fh * (0.14 + k * 0.05), cx + fw * 0.46, tipY - fh * 0.42)
        g.stroke()
      }
    }
  }

  // Fine loose-barb speckle, flowing with the plumage (downward), never
  // random-angled.
  for (let i = 0; i < 700; i++) {
    const x = hash(i * 3.1) * S
    const y = hash(i * 9.7) * S
    const a = 0.02 + hash(i * 5.3) * 0.04
    g.strokeStyle = `rgba(52, 44, 34, ${a.toFixed(3)})`
    g.lineWidth = 0.8
    g.beginPath()
    g.moveTo(x, y)
    g.lineTo(x + (hash(i * 1.9) - 0.5) * 4, y + 3 + hash(i * 2.7) * 4)
    g.stroke()
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.RepeatWrapping
  tex.anisotropy = 4
  return { map: tex, normalMap: makeNormalMap(canvas, 3.0, true) }
}

/**
 * Inject the feather-specific shading terms into a physical material:
 *
 * - a faint grazing-angle **rim light** in cool sky tones that separates the
 *   wing silhouette from the ground below. Deliberately tight (4th-power) and
 *   dim: the first-person wing is seen almost edge-on everywhere, so a broad
 *   or bright rim washes the whole surface toward sky tones and makes the
 *   wing read as translucent against terrain, and
 * - a per-vertex-gated **iridescent tint**: a view-angle-driven spectral
 *   sweep, enabled only where geometry bakes `aIri` > 0 (the rainbow
 *   lorikeet's green coverts), giving the oily green-to-blue feather shimmer.
 */
function injectFeatherShading(mat: THREE.MeshPhysicalMaterial): void {
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aIri;\nvarying float vIri;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvIri = aIri;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vIri;')
      .replace(
        '#include <emissivemap_fragment>',
        [
          '#include <emissivemap_fragment>',
          '{',
          '\tvec3 fpViewDir = normalize( vViewPosition );',
          '\tfloat fpNdV = saturate( dot( normal, fpViewDir ) );',
          '\tfloat fpRim = pow( 1.0 - fpNdV, 4.0 );',
          '\ttotalEmissiveRadiance += vec3( 0.018, 0.026, 0.038 ) * fpRim;',
          '\tvec3 fpIri = 0.5 + 0.5 * cos( 6.28318 * ( fpNdV * 1.25 + vec3( 0.0, 0.33, 0.67 ) ) );',
          '\ttotalEmissiveRadiance += fpIri * vIri * ( 0.02 + 0.05 * ( 1.0 - fpNdV ) );',
          '}',
        ].join('\n'),
      )
  }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/** Output record filled by a {@link makeGridGeometry} callback. */
interface GridVertex {
  x: number
  y: number
  z: number
  r: number
  g: number
  b: number
  u: number
  v: number
  /** Iridescence gate 0..1, written to the `aIri` attribute. */
  iri: number
}

/**
 * Build an indexed (cols × rows)-segment grid geometry. The callback fills
 * position, linear-space vertex colour, uv, and the iridescence gate for each
 * parametric (u, v). Normals are computed after filling.
 */
function makeGridGeometry(
  cols: number,
  rows: number,
  fill: (u: number, v: number, out: GridVertex) => void,
): THREE.BufferGeometry {
  const vertsX = cols + 1
  const vertsY = rows + 1
  const count = vertsX * vertsY
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  const iris = new Float32Array(count)
  const out: GridVertex = { x: 0, y: 0, z: 0, r: 1, g: 1, b: 1, u: 0, v: 0, iri: 0 }

  let p = 0
  let t = 0
  let q = 0
  for (let j = 0; j < vertsY; j++) {
    for (let i = 0; i < vertsX; i++) {
      out.iri = 0
      fill(i / cols, j / rows, out)
      positions[p] = out.x
      colors[p++] = out.r
      positions[p] = out.y
      colors[p++] = out.g
      positions[p] = out.z
      colors[p++] = out.b
      uvs[t++] = out.u
      uvs[t++] = out.v
      iris[q++] = out.iri
    }
  }

  const indices: number[] = []
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * vertsX + i
      const b = a + 1
      const c = a + vertsX
      const d = c + 1
      indices.push(a, c, b, b, c, d)
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  geo.setAttribute('aIri', new THREE.BufferAttribute(iris, 1))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  return geo
}

/**
 * A single flight feather. Extends outward along `side * +X` from its base at
 * the origin, with the vane in the X-Z plane, drooping slightly toward the
 * tip. The rounded silhouette comes from the alpha-cut feather texture. Root
 * AO is baked into the vertex colours so the quill visibly seats into the
 * plumage it grows from.
 */
function makeFeatherGeometry(
  length: number,
  width: number,
  side: 1 | -1,
  baseHex: number,
  tipHex: number,
  rootAO = 0.4,
  iri = 0,
): THREE.BufferGeometry {
  _cA.setHex(baseHex)
  _cB.setHex(tipHex)
  return makeGridGeometry(2, 7, (u, v, o) => {
    o.x = side * v * length
    o.y = -v * v * length * 0.12
    o.z = (u - 0.5) * width
    _cM.copy(_cA).lerp(_cB, smooth01(Math.pow(v, 1.4)))
    // Vane edges shade down toward the margins; no whitening multiplier —
    // brightened albedo on thin cards reads as edge piping at grazing angles.
    const edgeShade = 1 - Math.abs(u - 0.5) * 0.18
    const ao = lerp(1 - rootAO, 1, smooth01(v / 0.28))
    o.r = Math.min(1, _cM.r * edgeShade * ao)
    o.g = Math.min(1, _cM.g * edgeShade * ao)
    o.b = Math.min(1, _cM.b * edgeShade * ao)
    o.u = u
    o.v = v
    o.iri = iri * (1 - v)
  })
}

/** Species-specific plumage detail options for a wing panel and its cards. */
interface PanelOptions {
  /** Strength (0..1) of the baked wing-root AO that seats the wing. */
  rootAO: number
  /** Pale nape/covert band (wedge-tailed eagle). */
  napeBand?: { color: number; strength: number }
  /** Iridescence gate on the covert region (rainbow lorikeet). */
  iridescence?: number
}

/** Position scratch for {@link panelSurface} (init-time only). */
const _surf = { x: 0, y: 0, z: 0 }

/**
 * The shared parametric wing-panel surface: chord taper, forward sweep, and
 * a cambered aerofoil section. Both the base panel sheet and the covert
 * feather cards sample it so the cards ride exactly on the panel.
 */
function panelSurface(side: 1 | -1, span: number, chord: number, taper: number, sweep: number, u: number, v: number): void {
  const chordHere = chord * (1 - (1 - taper) * u)
  const zLead = -chordHere * 0.55 - u * span * sweep
  const zTrail = chordHere * 0.45 - u * span * sweep
  _surf.x = side * u * span
  _surf.z = lerp(zLead, zTrail, v)
  // Cambered aerofoil: mid-chord bulges up, tip droops slightly.
  _surf.y = Math.sin(v * Math.PI) * chord * 0.05 - u * u * span * 0.05
}

/**
 * A shaped wing panel (coverts + secondaries surface) spanning outward along
 * `side * +X`, with the chord along Z, subtle camber, forward sweep, tip
 * taper, and a scalloped trailing edge. Colour runs covert (leading edge)
 * into mid-wing (trailing edge) through a wide smoothstep gradient ramp — no
 * hard zone boundaries — with baked feather-row banding, wing-root AO, an
 * optional pale nape/covert band, and an optional iridescence gate.
 */
function makeWingPanelGeometry(
  side: 1 | -1,
  span: number,
  chord: number,
  taper: number,
  sweep: number,
  covertHex: number,
  midHex: number,
  opts: PanelOptions,
): THREE.BufferGeometry {
  _cA.setHex(covertHex)
  _cB.setHex(midHex)
  if (opts.napeBand) _cC.setHex(opts.napeBand.color)
  const iriAmount = opts.iridescence ?? 0
  return makeGridGeometry(7, 4, (u, v, o) => {
    panelSurface(side, span, chord, taper, sweep, u, v)
    o.x = _surf.x
    o.y = _surf.y
    o.z = _surf.z + v * Math.abs(Math.sin(u * Math.PI * 6.5)) * chord * 0.06
    // Covert → mid-wing gradient: a wide eased ramp instead of a hard seam.
    const blend = smooth01(v * 1.2 - 0.1)
    _cM.copy(_cA).lerp(_cB, blend)
    if (opts.napeBand) {
      // Pale covert band arcing across the upper wing, scalloped per feather
      // row so it reads as plumage, not paint.
      const scallop = 0.8 + 0.2 * Math.sin(u * Math.PI * 9 + v * 4)
      const bandMask = Math.exp(-Math.pow((v - 0.3) / 0.17, 2)) * scallop
      _cM.lerp(_cC, bandMask * opts.napeBand.strength)
    }
    const rowBand = 1 - 0.13 * Math.pow(Math.abs(Math.sin(v * Math.PI * 3)), 3)
    const speckle = 0.95 + hash(u * 61.7 + v * 13.9) * 0.1
    const tipShade = 1 - u * u * 0.16
    const rootAO = lerp(1 - opts.rootAO, 1, smooth01(u / 0.32))
    // Low-frequency AO where the covert card layers shingle over the sheet:
    // the leading half of the panel sits under the card stack and darkens,
    // grading back to full brightness on the exposed trailing edge.
    const underCards = 1 - 0.12 * (1 - smooth01(v * 1.6 - 0.5))
    const k = rowBand * speckle * tipShade * rootAO * underCards
    o.r = Math.min(1, _cM.r * k)
    o.g = Math.min(1, _cM.g * k)
    o.b = Math.min(1, _cM.b * k)
    o.u = u * span * 7
    o.v = v * chord * 7
    o.iri = iriAmount * (1 - blend) * smooth01(u * 3)
  })
}

/**
 * Layered covert feather cards for one wing panel: staggered rows of small
 * alpha-cut feather quads sampling {@link panelSurface}, each lifted slightly
 * and arched so the rows shingle over one another. This is what makes the
 * wing read as stacked feathers instead of a single textured plane. Card tips
 * take `tipHex` — the pale covert tips that band the eagle's upper wing.
 */
function makeCovertCardsGeometry(
  side: 1 | -1,
  span: number,
  chord: number,
  taper: number,
  sweep: number,
  covertHex: number,
  midHex: number,
  tipHex: number,
  rows: number,
  opts: PanelOptions,
): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = []
  const iriAmount = opts.iridescence ?? 0
  const cardsAcross = 7
  for (let r = 0; r < rows; r++) {
    const rowV = 0.1 + (r / rows) * 0.5
    const cardLenV = 0.36 + (r / rows) * 0.22
    for (let c = 0; c < cardsAcross; c++) {
      const seed = side * 13.7 + r * 17.7 + c * 5.3
      const jitter = hash(seed)
      const u0 = ((c + (r % 2 ? 0.5 : 0) + (jitter - 0.5) * 0.3) / cardsAcross) * 0.96 + 0.02
      const du = 0.16 + jitter * 0.03
      _cA.setHex(covertHex)
      _cB.setHex(midHex)
      _cD.setHex(tipHex)
      // Row base colour eases from covert toward mid-wing down the chord.
      _cA.lerp(_cB, smooth01(rowV * 1.2 - 0.1))
      const card = makeGridGeometry(1, 3, (cu, cv, o) => {
        const uu = Math.min(Math.max(u0 + (cu - 0.5) * du, 0), 1)
        const vv = Math.min(rowV + cv * cardLenV + (jitter - 0.5) * 0.05, 1)
        panelSurface(side, span, chord, taper, sweep, uu, vv)
        o.x = _surf.x
        o.z = _surf.z
        // Lift each row above the sheet; arch the card so its tip settles
        // back down over the next row — real shingling, with open edges the
        // rim light can catch.
        o.y = _surf.y + 0.004 + r * 0.0035 + Math.sin(cv * Math.PI) * chord * 0.045 * (0.6 + jitter * 0.5)
        _cM.copy(_cA).lerp(_cD, smooth01(cv * 1.6 - 0.7))
        const speckle = 0.92 + hash(seed + cu * 3.1 + cv * 7.9) * 0.13
        const rootAO = lerp(1 - opts.rootAO, 1, smooth01(uu / 0.32))
        // Low-frequency inter-layer AO: each card's base sits in the shadow
        // of the row shingled over it, so the stack reads as depth, not decals.
        const layerAO = lerp(0.68, 1, smooth01(cv * 1.7))
        o.r = Math.min(1, _cM.r * speckle * rootAO * layerAO)
        o.g = Math.min(1, _cM.g * speckle * rootAO * layerAO)
        o.b = Math.min(1, _cM.b * speckle * rootAO * layerAO)
        o.u = cu
        o.v = cv
        o.iri = iriAmount * (1 - rowV) * smooth01(uu * 3)
      })
      parts.push(card)
    }
  }
  const merged = mergeGeometries(parts, false)
  for (const part of parts) part.dispose()
  return merged
}

/**
 * A sculpted beak: elliptical cross-sections lofted forward along -Z, tapering
 * to a point, with a downward hooked culmen and a darker upper ridge.
 */
function makeBeakGeometry(style: BeakStyle): THREE.BufferGeometry {
  const ALONG = 10
  const AROUND = 12
  const count = (ALONG + 1) * (AROUND + 1)
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  _cA.setHex(style.baseColor)
  _cB.setHex(style.tipColor)

  let p = 0
  let t = 0
  for (let j = 0; j <= ALONG; j++) {
    const tj = j / ALONG
    const rx = Math.max(style.width * 0.5 * Math.pow(1 - tj, 0.72), 0.0012)
    const ry = Math.max(style.height * 0.5 * Math.pow(1 - tj, 0.6), 0.0012)
    const yDroop = -style.hook * tj * tj
    const z = -tj * style.length
    for (let i = 0; i <= AROUND; i++) {
      const a = (i / AROUND) * TWO_PI
      const sy = Math.sin(a)
      // Culmen (top) is fuller and rounder than the gonys (bottom).
      const yScale = sy > 0 ? 1.15 : 0.8
      positions[p] = Math.cos(a) * rx
      positions[p + 1] = yDroop + sy * ry * yScale
      positions[p + 2] = z
      _cM.copy(_cA).lerp(_cB, Math.pow(tj, 1.6))
      const ridgeShade = sy > 0 ? 1 - sy * 0.22 : 1 + sy * -0.04
      colors[p] = Math.min(1, _cM.r * ridgeShade)
      colors[p + 1] = Math.min(1, _cM.g * ridgeShade)
      colors[p + 2] = Math.min(1, _cM.b * ridgeShade)
      p += 3
      uvs[t++] = i / AROUND
      uvs[t++] = tj
    }
  }

  const indices: number[] = []
  for (let j = 0; j < ALONG; j++) {
    for (let i = 0; i < AROUND; i++) {
      const a = j * (AROUND + 1) + i
      const b = a + 1
      const c = a + AROUND + 1
      const d = c + 1
      indices.push(a, b, c, b, d, c)
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  return geo
}

/**
 * The soft breast shelf the bird sees along the bottom of its own view: a
 * domed sheet below and ahead of the eye, coloured with species breast tones
 * and subtle streak barring, with a fluffed near edge.
 */
function makeBreastGeometry(breastHex: number, streakHex: number): THREE.BufferGeometry {
  _cA.setHex(breastHex)
  _cB.setHex(streakHex)
  return makeGridGeometry(9, 5, (u, v, o) => {
    const across = (u - 0.5) * 0.36
    o.x = across
    o.z = -0.07 - v * 0.3
    // Dome: highest along the centreline, falling away to the sides and back,
    // with a soft feather-fluff ripple on the leading (near) edge.
    const dome = Math.cos((u - 0.5) * Math.PI * 1.15) * 0.05
    const ripple = (1 - v) * Math.sin(u * Math.PI * 9) * 0.004
    o.y = -0.165 + dome * (1 - v * 0.55) - v * 0.055 + ripple
    const barring = Math.pow(Math.abs(Math.sin(v * Math.PI * 4 + u * 2.4)), 6)
    _cM.copy(_cA).lerp(_cB, barring * 0.55 + hash(u * 47.1 + v * 19.3) * 0.12)
    // Ambient-shade the sides so the dome reads as a volume, and darken the
    // far corners where the wing roots shadow the shoulders.
    const wingShadow = 1 - smooth01(v * 1.4 - 0.5) * smooth01(Math.abs(u - 0.5) * 3 - 0.5) * 0.3
    const shade = (0.82 + Math.cos((u - 0.5) * Math.PI) * 0.18) * wingShadow
    o.r = Math.min(1, _cM.r * shade)
    o.g = Math.min(1, _cM.g * shade)
    o.b = Math.min(1, _cM.b * shade)
    o.u = u * 3
    o.v = v * 3
  })
}

/**
 * A scapular tuft — a small feathered dome parented at the wing shoulder so
 * the articulated wing visually emerges from a plumage mass instead of a bare
 * joint. Colour blends breast into covert with speckle.
 */
function makeScapularGeometry(side: 1 | -1, size: number, breastHex: number, covertHex: number): THREE.BufferGeometry {
  _cA.setHex(breastHex)
  _cB.setHex(covertHex)
  return makeGridGeometry(6, 4, (u, v, o) => {
    // u wraps around the shoulder (inboard → outboard), v runs front → back.
    const across = (u - 0.5) * size * 1.5
    const along = (v - 0.5) * size * 1.9
    const domeNorm = Math.cos((u - 0.5) * Math.PI * 0.9) * Math.cos((v - 0.5) * Math.PI * 0.85)
    const dome = domeNorm * size * 0.42
    o.x = side * (across + size * 0.2)
    o.y = dome - size * 0.28
    o.z = along
    _cM.copy(_cA).lerp(_cB, smooth01(u * 1.3))
    const fluff = 0.9 + hash(u * 33.1 + v * 21.7) * 0.16
    // Contact AO around the tuft rim seats it into the body plumage.
    const ao = 0.72 + 0.28 * domeNorm
    o.r = Math.min(1, _cM.r * fluff * ao)
    o.g = Math.min(1, _cM.g * fluff * ao)
    o.b = Math.min(1, _cM.b * fluff * ao)
    o.u = u * 2
    o.v = v * 2
  })
}

/** Bake a solid linear-space colour attribute onto an existing geometry. */
function addSolidColor(geo: THREE.BufferGeometry, hex: number): void {
  _cA.setHex(hex)
  const count = geo.getAttribute('position').count
  const colors = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    colors[i * 3] = _cA.r
    colors[i * 3 + 1] = _cA.g
    colors[i * 3 + 2] = _cA.b
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
}

/**
 * A complete low-poly ambient bird (body, tail, both wings) merged into one
 * geometry, facing -Z. After merging, an `aFlap` attribute (0 at the body,
 * 1 at the wingtip) is derived from |x| so a vertex shader can flap the wings
 * for free.
 */
function makeAmbientBirdGeometry(
  bodyLength: number,
  halfSpan: number,
  bodyHex: number,
  wingHex: number,
  tipHex: number,
  dihedral: number,
): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = []

  const body = new THREE.SphereGeometry(0.5, 8, 6)
  body.scale(bodyLength * 0.34, bodyLength * 0.3, bodyLength)
  addSolidColor(body, bodyHex)
  parts.push(body)

  const tail = new THREE.PlaneGeometry(bodyLength * 0.34, bodyLength * 0.55, 1, 1)
  tail.rotateX(-Math.PI / 2)
  tail.translate(0, 0.01, bodyLength * 0.68)
  addSolidColor(tail, wingHex)
  parts.push(tail)

  _cA.setHex(wingHex)
  _cB.setHex(tipHex)
  for (const side of [-1, 1] as const) {
    const wing = makeGridGeometry(5, 1, (u, v, o) => {
      const chord = bodyLength * 0.62 * (1 - 0.62 * u)
      o.x = side * (bodyLength * 0.12 + u * halfSpan)
      o.y = u * halfSpan * dihedral
      o.z = (v - 0.5) * chord + u * u * halfSpan * 0.3
      _cM.copy(_cA).lerp(_cB, u * u)
      o.r = _cM.r
      o.g = _cM.g
      o.b = _cM.b
      o.u = u
      o.v = v
    })
    // The ambient material has no iridescence injection; drop the gate
    // attribute so every merged part carries an identical attribute set.
    wing.deleteAttribute('aIri')
    parts.push(wing)
  }

  const merged = mergeGeometries(parts, false)
  for (const part of parts) part.dispose()

  const pos = merged.getAttribute('position')
  const flap = new Float32Array(pos.count)
  const inner = bodyLength * 0.14
  for (let i = 0; i < pos.count; i++) {
    const raw = (Math.abs(pos.getX(i)) - inner) / (halfSpan - inner)
    flap[i] = Math.pow(Math.min(Math.max(raw, 0), 1), 1.35)
  }
  merged.setAttribute('aFlap', new THREE.BufferAttribute(flap, 1))
  return merged
}

// ---------------------------------------------------------------------------
// First-person rig structures
// ---------------------------------------------------------------------------

/** One articulated wing: shoulder → elbow → wrist bone chain plus feathers. */
interface WingRig {
  side: 1 | -1
  shoulder: THREE.Group
  elbow: THREE.Group
  wrist: THREE.Group
  primaries: THREE.Mesh[]
  /** Base fan angle (radians about Y) for each primary. */
  primaryFan: number[]
  secondaries: THREE.Mesh[]
  /** Resting droop angle (radians about X) for each secondary. */
  secondaryDroop: number[]
}

/** A complete first-person body kit for one species. */
interface SpeciesRig {
  root: THREE.Group
  wings: [WingRig, WingRig]
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/**
 * The Birds game module: first-person per-species bird rigs attached to the
 * camera, plus ambient wildlife (a corella skein and soaring raptors).
 * See the file header for the full design.
 */
export class Birds implements GameModule {
  readonly name = 'birds'

  /** Camera-parented container for every species rig. */
  private fpRoot!: THREE.Group
  private rigs!: Map<SpeciesId, SpeciesRig>
  private activeSpecies: SpeciesId | null = null

  // Shared feather materials (physical: sheen + anisotropic specular, with
  // injected rim light and gated iridescence).
  private featherMat!: THREE.MeshPhysicalMaterial
  private plumageMat!: THREE.MeshPhysicalMaterial
  private beakMat!: THREE.MeshStandardMaterial

  // Ambient corella flock.
  private flock!: THREE.InstancedMesh
  private readonly flockTime = { value: 0 }
  /** Per-bird lateral/vertical offsets from its own path point. */
  private flockOffsets!: Float32Array
  /** Per-bird path-angle lag, strings the flock into a skein. */
  private flockLag!: Float32Array
  private flockSwirl!: Float32Array
  private flockPhase!: Float32Array
  private flockYawJitter!: Float32Array
  private flockAltitude = 140
  private flockLastHeading = 0
  private flockRoll = 0
  private flockHeadingInit = false

  // Soaring raptors.
  private raptors: THREE.Mesh[] = []
  private readonly raptorOrbits = [
    { cx: -3350, cz: -3250, radius: 270, omega: 0.052, phase: 0.0, height: 380, altitude: 900 },
    { cx: -2650, cz: -3750, radius: 340, omega: -0.044, phase: 2.4, height: 320, altitude: 800 },
  ]

  /** Build textures, materials, all five species rigs, and the ambient life. */
  init(ctx: GameContext): void {
    const feather = makeFeatherTexture()
    const plumage = makePlumageTexture()

    // Both wing materials render in the opaque pass with full depth writes —
    // alpha testing only cuts the feather-card silhouettes. Specular, sheen,
    // and rim are all kept low: the first-person wing is viewed almost
    // edge-on everywhere, and any strong grazing-angle response desaturates
    // the plumage toward sky/terrain values, which reads on screen as a
    // ghost-transparent wing even though the surface is fully opaque.
    this.featherMat = new THREE.MeshPhysicalMaterial({
      map: feather.map,
      normalMap: feather.normalMap,
      normalScale: new THREE.Vector2(0.65, 0.65),
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: false,
      depthWrite: true,
      alphaTest: 0.32,
      roughness: 0.7,
      metalness: 0,
      // Low-gloss anisotropic specular: the highlight streaks along the barbs
      // the way light runs across a real feather vane, without laminate gleam.
      specularIntensity: 0.16,
      anisotropy: 0.6,
      anisotropyRotation: Math.PI / 2,
      sheen: 0.15,
      sheenRoughness: 0.65,
      sheenColor: new THREE.Color(0xe6d9c4),
    })
    this.plumageMat = new THREE.MeshPhysicalMaterial({
      map: plumage.map,
      normalMap: plumage.normalMap,
      normalScale: new THREE.Vector2(0.5, 0.5),
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: false,
      depthWrite: true,
      roughness: 0.85,
      metalness: 0,
      specularIntensity: 0.1,
      anisotropy: 0.4,
      sheen: 0.14,
      sheenRoughness: 0.7,
      sheenColor: new THREE.Color(0xe6d9c4),
    })
    injectFeatherShading(this.featherMat)
    injectFeatherShading(this.plumageMat)
    this.beakMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.34,
      metalness: 0.02,
    })

    this.fpRoot = new THREE.Group()
    this.fpRoot.name = 'birds:first-person'
    ctx.camera.add(this.fpRoot)

    this.rigs = new Map()
    for (const species of Object.keys(SPECIES_STYLE) as SpeciesId[]) {
      const rig = this.buildSpeciesRig(SPECIES_STYLE[species])
      rig.root.visible = false
      this.fpRoot.add(rig.root)
      this.rigs.set(species, rig)
    }
    this.setActiveSpecies(ctx.species)

    this.buildFlock(ctx)
    this.buildRaptors(ctx)
  }

  /** Animate the first-person wings, the corella flock, and the raptors. */
  update(dt: number, ctx: GameContext): void {
    if (ctx.species !== this.activeSpecies) this.setActiveSpecies(ctx.species)
    this.fpRoot.visible = ctx.flying

    if (this.fpRoot.visible && this.activeSpecies) {
      const rig = this.rigs.get(this.activeSpecies)
      if (rig) this.animateFirstPerson(rig, ctx)
    }

    this.animateFlock(dt, ctx)
    this.animateRaptors(ctx)
  }

  // -------------------------------------------------------------------------
  // First-person rig construction
  // -------------------------------------------------------------------------

  /** Build the full camera-space body kit for one species. */
  private buildSpeciesRig(style: SpeciesStyle): SpeciesRig {
    const root = new THREE.Group()
    const wings: WingRig[] = []
    for (const side of [-1, 1] as const) wings.push(this.buildWing(side, style, root))

    const beak = new THREE.Mesh(makeBeakGeometry(style.beak), this.beakMat)
    beak.position.set(0, -0.07, -0.05)
    beak.rotation.x = -0.3
    root.add(beak)

    const breast = new THREE.Mesh(makeBreastGeometry(style.breast, style.breastStreak), this.plumageMat)
    root.add(breast)

    root.traverse((obj) => {
      obj.frustumCulled = false
      if ((obj as THREE.Mesh).isMesh) {
        obj.castShadow = false
        obj.receiveShadow = false
      }
    })
    return { root, wings: [wings[0], wings[1]] as [WingRig, WingRig] }
  }

  /** Build one articulated wing and parent it under `root`. */
  private buildWing(side: 1 | -1, style: SpeciesStyle, root: THREE.Group): WingRig {
    const H = style.halfSpan
    const spanIn = H * 0.3
    const spanOut = H * 0.28
    const chordIn = H * 0.3 + 0.02
    const chordOut = H * 0.24 + 0.015

    const shoulder = new THREE.Group()
    shoulder.position.set(side * (0.03 + H * 0.05), -0.06 - H * 0.03, -0.09 - H * 0.1)
    root.add(shoulder)

    // Scapular tuft covers the wing-root joint so the wing grows out of
    // plumage. It is parented to the shoulder and rides the stroke.
    const tuft = new THREE.Mesh(
      makeScapularGeometry(side, H * 0.16 + 0.03, style.breast, style.covert),
      this.plumageMat,
    )
    shoulder.add(tuft)

    const innerOpts: PanelOptions = { rootAO: 0.5, napeBand: style.napeBand, iridescence: style.iridescence }
    const outerOpts: PanelOptions = { rootAO: 0.32, napeBand: style.napeBand, iridescence: style.iridescence }

    const innerPanel = new THREE.Mesh(
      makeWingPanelGeometry(side, spanIn, chordIn, 0.9, 0.28, style.covert, style.midWing, innerOpts),
      this.plumageMat,
    )
    shoulder.add(innerPanel)

    // Layered covert feather cards shingle over the panel sheets so the wing
    // surface reads as stacked feathers, with the species' covert-tip banding.
    const innerCards = new THREE.Mesh(
      makeCovertCardsGeometry(side, spanIn, chordIn, 0.9, 0.28, style.covert, style.midWing, style.covertTip, 3, innerOpts),
      this.featherMat,
    )
    shoulder.add(innerCards)

    const elbow = new THREE.Group()
    elbow.position.set(side * spanIn, -spanIn * 0.05, -spanIn * 0.28)
    shoulder.add(elbow)

    const outerPanel = new THREE.Mesh(
      makeWingPanelGeometry(side, spanOut, chordOut, 0.72, 0.16, style.covert, style.midWing, outerOpts),
      this.plumageMat,
    )
    elbow.add(outerPanel)

    const outerCards = new THREE.Mesh(
      makeCovertCardsGeometry(side, spanOut, chordOut, 0.72, 0.16, style.covert, style.midWing, style.covertTip, 2, outerOpts),
      this.featherMat,
    )
    elbow.add(outerCards)

    const wrist = new THREE.Group()
    wrist.position.set(side * spanOut, -spanOut * 0.04, -spanOut * 0.16)
    elbow.add(wrist)

    // Primary feathers fanning from the wrist ("fingers"). Pointed wings
    // (lorikeet, galah) sweep the fan back and lengthen the leading feather;
    // broad slotted wings (eagle) keep the fan wide and even.
    const primaries: THREE.Mesh[] = []
    const primaryFan: number[] = []
    const n = style.primaryCount
    const maxPrimaryLen = H * 0.44 + 0.03
    for (let i = 0; i < n; i++) {
      const t = n > 1 ? i / (n - 1) : 0
      const taperCurve = lerp(0.5 * Math.pow(t, 1.25), 0.72 * t * t, style.pointedness)
      const len = maxPrimaryLen * (1 + style.pointedness * 0.18) * (1 - taperCurve)
      const width = (H * 0.055 + 0.012) * (1 + t * 0.35) * (1 - style.pointedness * 0.25)
      const feather = new THREE.Mesh(
        makeFeatherGeometry(len, width, side, style.primary, style.primaryTip),
        this.featherMat,
      )
      feather.position.set(side * t * 0.012, -t * 0.002, t * 0.01)
      wrist.add(feather)
      primaries.push(feather)
      primaryFan.push(-0.14 + t * lerp(1.05, 0.7, style.pointedness))
    }

    // Secondary feathers hung off the panel trailing edges.
    const secondaries: THREE.Mesh[] = []
    const secondaryDroop: number[] = []
    const hangSecondary = (parent: THREE.Group, span: number, chord: number, count: number, seed: number): void => {
      for (let i = 0; i < count; i++) {
        const t = (i + 0.5) / count
        const len = chord * 0.85 + H * 0.06
        const feather = new THREE.Mesh(
          makeFeatherGeometry(len, H * 0.07 + 0.014, 1, style.midWing, style.primaryTip),
          this.featherMat,
        )
        // Secondaries point backward (+Z): rotate the outward feather 90°.
        feather.rotation.y = -side * (Math.PI / 2)
        feather.position.set(side * t * span, -0.004, chord * 0.3 - t * span * 0.22)
        parent.add(feather)
        secondaries.push(feather)
        secondaryDroop.push(0.12 + hash(seed + i * 3.3) * 0.1)
      }
    }
    hangSecondary(shoulder, spanIn, chordIn, 4, side * 7)
    hangSecondary(elbow, spanOut, chordOut, 3, side * 31)

    return { side, shoulder, elbow, wrist, primaries, primaryFan, secondaries, secondaryDroop }
  }

  /** Show the rig for `species`, hiding the others. */
  private setActiveSpecies(species: SpeciesId): void {
    for (const [id, rig] of this.rigs) rig.root.visible = id === species
    this.activeSpecies = species
  }

  // -------------------------------------------------------------------------
  // First-person animation
  // -------------------------------------------------------------------------

  /**
   * Waveform of one wing joint at `phase` (0..1) with a phase `lag`.
   * Returns +1 with the wing raised and -1 at the bottom of the downstroke;
   * the downstroke occupies {@link DOWNSTROKE_FRACTION} of the cycle so it
   * reads fast and powerful against the relaxed upstroke.
   */
  private strokeAt(phase: number, lag: number): number {
    let p = phase - lag
    p -= Math.floor(p)
    const shaped =
      p < DOWNSTROKE_FRACTION
        ? (p / DOWNSTROKE_FRACTION) * 0.5
        : 0.5 + ((p - DOWNSTROKE_FRACTION) / (1 - DOWNSTROKE_FRACTION)) * 0.5
    return Math.cos(shaped * TWO_PI)
  }

  /** Pose both wings, the body bob, and the feather micro-motion. */
  private animateFirstPerson(rig: SpeciesRig, ctx: GameContext): void {
    const f = ctx.flight
    const t = ctx.time
    const k = f.flapStrength
    const glide = 1 - k
    const wind = Math.min(1, f.speed / 28)

    const stroke = this.strokeAt(f.flapPhase, 0)
    const strokeE = this.strokeAt(f.flapPhase, ELBOW_LAG)
    const strokeW = this.strokeAt(f.flapPhase, WRIST_LAG)
    const upFold = Math.max(0, strokeE)
    const twist = -Math.sin((f.flapPhase - WRIST_LAG) * TWO_PI) * 0.28 * k

    // Whole-body life: heave with the downstroke, gaze-stabilising
    // counter-roll against the bank, shudder in a stall.
    rig.root.position.y = -0.004 + stroke * 0.01 * k
    rig.root.rotation.x = strokeW * 0.02 * k
    rig.root.rotation.z = -f.banking * 0.07 + (f.stalled ? Math.sin(t * 47) * 0.012 : 0)

    for (const wing of rig.wings) {
      const side = wing.side
      const shudder = f.stalled ? Math.sin(t * 46 + side * 1.3) * 0.06 : 0
      // Inside wing tucks slightly through a banked turn; the outside wing
      // lifts a touch, holding the turn.
      const bankHere = f.banking * side
      const insideTuck = Math.max(0, bankHere) * 0.2
      const outsideLift = Math.max(0, -bankHere) * 0.07

      const shoulderLift = (0.12 + outsideLift + 0.05 * Math.sin(t * 1.7 + side)) * glide + (0.18 + stroke * 0.95) * k
      wing.shoulder.rotation.z = side * (shoulderLift + shudder)
      wing.shoulder.rotation.x = strokeW * 0.09 * k

      wing.elbow.rotation.z = side * (-0.05 * glide + strokeE * 0.45 * k)
      wing.elbow.rotation.y = -side * (upFold * 0.5 * k + 0.05 * glide + insideTuck * 0.4)

      wing.wrist.rotation.z = side * (0.04 * glide + strokeW * 0.55 * k)
      wing.wrist.rotation.y = -side * (upFold * 0.75 * k + 0.08 * glide + insideTuck)
      wing.wrist.rotation.x = twist

      // Primary fan: fingers spread wide in a glide, sweep shut as the wing
      // folds through the upstroke; each feather flutters in the airstream.
      const spread = 0.55 + 0.45 * glide - 0.3 * upFold * k
      for (let i = 0; i < wing.primaries.length; i++) {
        const feather = wing.primaries[i]
        feather.rotation.y = -side * wing.primaryFan[i] * spread
        feather.rotation.x = twist * 0.35 + Math.sin(t * 22 + i * 1.9 + side * 3) * 0.02 * wind
      }
      for (let i = 0; i < wing.secondaries.length; i++) {
        wing.secondaries[i].rotation.x =
          wing.secondaryDroop[i] + Math.sin(t * 17 + i * 2.3 + side * 5) * 0.016 * wind
      }
    }
  }

  // -------------------------------------------------------------------------
  // Ambient life
  // -------------------------------------------------------------------------

  /** Build the instanced little-corella flock with shader-driven flapping. */
  private buildFlock(ctx: GameContext): void {
    const geo = makeAmbientBirdGeometry(0.42, 0.38, 0xf4efe4, 0xece6d6, 0xd6cfbc, 0.06)

    const phases = new Float32Array(FLOCK_COUNT)
    this.flockOffsets = new Float32Array(FLOCK_COUNT * 3)
    this.flockLag = new Float32Array(FLOCK_COUNT)
    this.flockSwirl = new Float32Array(FLOCK_COUNT)
    this.flockPhase = phases
    this.flockYawJitter = new Float32Array(FLOCK_COUNT)
    for (let i = 0; i < FLOCK_COUNT; i++) {
      // Skein shape: each bird trails the lead by a slice of path angle, with
      // a lateral echelon offset that widens down the string.
      const rank = i / FLOCK_COUNT
      this.flockLag[i] = rank * 0.11 + (hash(i * 23.9) - 0.5) * 0.012
      const echelon = (i % 2 === 0 ? 1 : -1) * (2 + rank * 26)
      this.flockOffsets[i * 3] = echelon + (hash(i * 3.7) - 0.5) * 10
      this.flockOffsets[i * 3 + 1] = (hash(i * 7.1) - 0.5) * 14
      this.flockOffsets[i * 3 + 2] = (hash(i * 11.9) - 0.5) * 12
      this.flockSwirl[i] = 0.25 + hash(i * 5.3) * 0.35
      phases[i] = hash(i * 13.7) * TWO_PI
      this.flockYawJitter[i] = (hash(i * 17.3) - 0.5) * 0.3
    }
    geo.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1))

    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide })
    const uniform = this.flockTime
    mat.onBeforeCompile = (shader) => {
      shader.uniforms['uTime'] = uniform
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          '#include <common>\nattribute float aFlap;\nattribute float aPhase;\nuniform float uTime;',
        )
        .replace(
          '#include <begin_vertex>',
          [
            '#include <begin_vertex>',
            // Flap-and-glide envelope: each bird beats in bursts, then rests.
            'float birdEnv = 0.35 + 0.65 * smoothstep(-0.2, 0.6, sin(uTime * 0.85 + aPhase * 5.0));',
            'transformed.y += sin(uTime * 11.0 + aPhase) * aFlap * aFlap * 0.3 * birdEnv;',
          ].join('\n'),
        )
    }

    this.flock = new THREE.InstancedMesh(geo, mat, FLOCK_COUNT)
    this.flock.name = 'birds:corella-flock'
    this.flock.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.flock.frustumCulled = false
    this.flock.castShadow = false
    this.flock.receiveShadow = false
    // Faint warm/cool tint variation so the flock does not read as clones.
    for (let i = 0; i < FLOCK_COUNT; i++) {
      const warm = hash(i * 41.3)
      _cM.setRGB(0.96 + warm * 0.04, 0.955 + warm * 0.035, 0.94 + (1 - warm) * 0.05)
      this.flock.setColorAt(i, _cM)
    }
    ctx.scene.add(this.flock)
  }

  /** Build the two far soaring raptor silhouettes over the ranges. */
  private buildRaptors(ctx: GameContext): void {
    const geo = makeAmbientBirdGeometry(0.95, 1.15, 0x241a10, 0x1c140b, 0x110d07, 0.11)
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, side: THREE.DoubleSide })
    for (let i = 0; i < this.raptorOrbits.length; i++) {
      const raptor = new THREE.Mesh(geo, mat)
      raptor.name = `birds:raptor-${i}`
      raptor.rotation.order = 'YXZ'
      raptor.frustumCulled = false
      raptor.castShadow = false
      raptor.receiveShadow = false
      ctx.scene.add(raptor)
      this.raptors.push(raptor)
    }
  }

  /** X of the corella loop at path angle `a`. */
  private flockPathX(a: number): number {
    return FLOCK_ANCHOR_X + Math.cos(a) * FLOCK_RADIUS_X + Math.sin(a * 2.1) * 160
  }

  /** Z of the corella loop at path angle `a`. */
  private flockPathZ(a: number): number {
    return FLOCK_ANCHOR_Z + Math.sin(a) * FLOCK_RADIUS_Z + Math.cos(a * 1.7) * 140
  }

  /** Move the corella skein along its valley loop and pose every instance. */
  private animateFlock(dt: number, ctx: GameContext): void {
    const t = ctx.time
    this.flockTime.value = t

    const a = t * 0.045
    const cx = this.flockPathX(a)
    const cz = this.flockPathZ(a)
    const dxda = -Math.sin(a) * FLOCK_RADIUS_X + Math.cos(a * 2.1) * 2.1 * 160
    const dzda = Math.cos(a) * FLOCK_RADIUS_Z - Math.sin(a * 1.7) * 1.7 * 140
    const heading = Math.atan2(-dxda, -dzda)
    const sinH = Math.sin(heading)
    const cosH = Math.cos(heading)

    // Smoothed terrain-following altitude, sampled at the lead bird.
    const ground = Math.max(ctx.getTerrainHeight(cx, cz), 0)
    this.flockAltitude += (ground + FLOCK_CLEARANCE - this.flockAltitude) * Math.min(1, dt * 0.5)

    // Bank into turns: smoothed heading rate drives the flock roll.
    if (!this.flockHeadingInit) {
      this.flockLastHeading = heading
      this.flockHeadingInit = true
    }
    let dHeading = heading - this.flockLastHeading
    dHeading -= Math.round(dHeading / TWO_PI) * TWO_PI
    this.flockLastHeading = heading
    if (dt > 0) {
      const targetRoll = Math.max(-0.5, Math.min(0.5, (dHeading / dt) * 3))
      this.flockRoll += (targetRoll - this.flockRoll) * Math.min(1, dt * 3)
    }

    for (let i = 0; i < FLOCK_COUNT; i++) {
      // Each bird flies its own point on the path, trailing the lead — the
      // flock stretches into a line through turns like a real corella skein.
      const ai = a - this.flockLag[i]
      const px = this.flockPathX(ai)
      const pz = this.flockPathZ(ai)
      const swirl = t * this.flockSwirl[i] + this.flockPhase[i]
      // Echelon offset is applied across the heading so the V holds shape.
      const across = this.flockOffsets[i * 3] + Math.sin(swirl) * 5
      _pos.set(
        px + cosH * across,
        this.flockAltitude + this.flockOffsets[i * 3 + 1] + Math.sin(swirl * 0.63 + 1.3) * 4,
        pz - sinH * across + this.flockOffsets[i * 3 + 2] + Math.cos(swirl) * 4,
      )
      _euler.set(0, heading + this.flockYawJitter[i], this.flockRoll, 'YXZ')
      _quat.setFromEuler(_euler)
      _mat.compose(_pos, _quat, _unitScale)
      this.flock.setMatrixAt(i, _mat)
    }
    this.flock.instanceMatrix.needsUpdate = true
  }

  /** Fly the raptor silhouettes in slow banked thermal circles. */
  private animateRaptors(ctx: GameContext): void {
    const t = ctx.time
    for (let i = 0; i < this.raptors.length; i++) {
      const orbit = this.raptorOrbits[i]
      const raptor = this.raptors[i]
      const a = t * orbit.omega + orbit.phase
      const x = orbit.cx + Math.cos(a) * orbit.radius
      const z = orbit.cz + Math.sin(a) * orbit.radius
      // Ease the tracked altitude toward the local thermal ceiling.
      const target = Math.max(ctx.getTerrainHeight(x, z), 0) + orbit.height
      orbit.altitude += (target - orbit.altitude) * 0.01
      raptor.position.set(x, orbit.altitude + Math.sin(t * 0.11 + i * 2.1) * 14, z)

      const dir = orbit.omega >= 0 ? 1 : -1
      raptor.rotation.y = Math.atan2(Math.sin(a) * dir, -Math.cos(a) * dir)
      raptor.rotation.x = Math.sin(t * 0.5 + i) * 0.03
      raptor.rotation.z = -0.26 * dir + Math.sin(t * 0.7 + i * 1.7) * 0.05
    }
  }
}
