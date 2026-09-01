/**
 * Vegetation — eucalypt woodland, spinifex grass and granite boulders.
 *
 * Everything is procedural and deterministic:
 * - Four gum-tree archetypes (river red gum, ghost gum, stringybark, mulga)
 *   are grown once at init by a recursive branch builder. Trunk + branches are
 *   tapered tubes with baked bark vertex colors; the canopy is a cloud of
 *   leaf-cluster cards cut from a canvas-painted atlas (alpha-tested).
 * - Trees are placed by height / slope / moisture masks driven by the terrain
 *   sampler, PLUS a terrain-concavity (Laplacian) term so stands chase gully
 *   floors and drainage lines, and a Poisson-disc spacing gate whose radius
 *   follows the copse noise — tight crown spacing in copse cores, wide in
 *   open savanna — so the forest reads as clustered woodland, never uniform
 *   speckle. Trees are bucketed into 750 m chunks that mirror the terrain
 *   grid. Near
 *   chunks render full-geometry InstancedMeshes (2 draw calls each: bark +
 *   foliage); far trees render as ONE global instanced impostor mesh with
 *   three LOD card sets — fixed world-space cross cards mid-range, a
 *   cylindrical billboard far, plus an overhead canopy card that fades in
 *   as the camera looks down (aerial views see round crowns, never edge-on
 *   slivers) — cut from a sun-shaded silhouette atlas (two variants per
 *   archetype) that mirrors each card so its baked lit side always faces
 *   the real sun. Coverage uses a solid alpha cut; dithering applies ONLY
 *   inside the near/far hand-over band, so steady-state distant trees never
 *   sparkle into salt-and-pepper speckle.
 *   The near/far hand-over uses the SAME chunk-center distance test on the
 *   CPU and in the impostor shader, so coverage never gaps — and BOTH sides
 *   add the same live altitude boost, so climbing shrinks the full-geometry
 *   ring and hands the whole midground to the anti-speckle impostor crowns.
 * - Every tree also drops a soft elliptical blob-shadow decal (one instanced
 *   sheet sharing the impostor's attribute buffers) that stretches away from
 *   the live sun azimuth, grounding trees far beyond the real shadow-map
 *   radius so they never float on the terrain.
 * - Tree tint/hue comes from a LOW-FREQUENCY stand-color noise field (plus a
 *   whisper of per-tree jitter), so neighbouring trees share a palette and
 *   whole hillsides drift warm/cool instead of a per-tree color lottery.
 * - Grass/spinifex tussocks are crossed alpha-tested cards in a single
 *   InstancedMesh ring around the camera, re-scattered from a world-space
 *   cell hash whenever the camera strays, so placement is stable in space.
 *   A vertex shader sways them in the wind and shrinks them to nothing at
 *   the ring edge (density fade).
 * - A second, wider scrub ring (saltbush / spinifex shrubs, ~330 m) fills the
 *   ground between trees with parallax detail at low altitude. Density is
 *   clump-noise + moisture driven, and instances shrink out at the ring edge
 *   exactly like the grass.
 * - Granite boulders are two noise-displaced rock meshes, globally instanced,
 *   speckled and lichen-stained in the fragment shader.
 *
 * Lighting: foliage and ground-cover materials add a sun-wrap + back-scatter
 * translucency term (shared `uSunDir` uniform, fed from ctx.sunDirection) so
 * low sun glows THROUGH canopies instead of leaving their shadow side black;
 * the far impostors carry the matching per-vertex backlight rim.
 *
 * Wind: one shared `uTime` uniform (fed from ctx.time) drives a gust field
 * evaluated in world space, applied to trunk tips, canopy cards and grass in
 * the vertex shader — and identically in the shadow-depth materials so the
 * shadows move with the trees.
 *
 * Per-frame cost: a 256-iteration visibility walk over chunk records, a few
 * uniform copies, and (rarely) a grass ring re-scatter. Zero steady-state
 * allocation.
 */
import * as THREE from 'three'
import { createNoise2D, createNoise3D } from 'simplex-noise'
import { WORLD, type GameContext, type GameModule } from '../core/GameState'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Half the world edge, meters. */
const HALF = WORLD.size / 2
/** Vegetation chunks per side (matches the terrain chunk grid). */
const CHUNKS = 16
/** World size of one vegetation chunk, meters. */
const CHUNK_SIZE = WORLD.size / CHUNKS
/** Horizontal chunk-center distance inside which full tree geometry renders. */
const TREE_NEAR = 1250
/** Impostor cross-fade band (fully faded in at {@link TREE_NEAR}). */
const IMPOSTOR_FADE = 90
/** Chunk-center distance where cross-card impostors hand over to the far billboard. */
const IMPOSTOR_CROSS = 2600
/** Width of the cross-card → billboard hand-over band, meters. */
const IMPOSTOR_CROSS_FADE = 700
/** Horizontal chunk-center distance inside which trees cast shadows. */
const TREE_SHADOW = 900
/** Grass ring radius, meters. */
const GRASS_RADIUS = 480
/** Grass scatter cell size, meters (one tussock candidate per cell). */
const GRASS_CELL = 12
/** Re-scatter the grass ring when the camera drifts this far, meters. */
const GRASS_RESEED = 48
/** Hide the grass mesh entirely above this altitude over terrain, meters. */
const GRASS_MAX_ALT = 700
/** Maximum live tussock instances. */
const GRASS_CAPACITY = 5200
/** Scrub (shrub billboard) ring radius, meters. */
const SCRUB_RADIUS = 360
/** Scrub scatter cell size, meters. */
const SCRUB_CELL = 11
/** Hide the scrub mesh above this altitude over terrain, meters. */
const SCRUB_MAX_ALT = 950
/** Maximum live scrub instances. */
const SCRUB_CAPACITY = 4200
/** Camera altitude over terrain where the tree LOD rings start shrinking. */
const ALT_LOD_START = 120
/** LOD-ring shrink, meters per meter of altitude beyond {@link ALT_LOD_START}. */
const ALT_LOD_GAIN = 2.0
/** Grass altitude scale-fade start, meters over terrain (ends at {@link GRASS_MAX_ALT}). */
const GRASS_ALT_FADE = 400
/** Scrub altitude scale-fade start, meters over terrain (ends at {@link SCRUB_MAX_ALT}). */
const SCRUB_ALT_FADE = 560
/** Candidate points thrown at each chunk for tree placement. */
const TREE_CANDIDATES = 640
/** Hard cap on trees accepted per chunk — bounds the near-geometry budget. */
const TREE_CHUNK_CAP = 250
/** Candidate points thrown at each chunk for boulder placement. */
const ROCK_CANDIDATES = 26

// ---------------------------------------------------------------------------
// Deterministic RNG + math helpers
// ---------------------------------------------------------------------------

/** Mulberry32 PRNG — fixed seeds keep every boot's forest identical. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 2-D integer hash → 0..1. Stable scatter key for world-space cells. */
function hash2(ix: number, iz: number): number {
  let h = (ix * 374761393 + iz * 668265263) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

/** Clamped Hermite smoothstep, matching GLSL semantics. */
function sstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)))
  return t * t * (3 - 2 * t)
}

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Clamp to 0..1. */
function sat(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

// ---------------------------------------------------------------------------
// Shared wind shader code
// ---------------------------------------------------------------------------

/** GLSL gust field: world-space position + time → horizontal sway offset. */
const WIND_GLSL = /* glsl */ `
uniform float uTime;
attribute float aSway;
vec2 avWind( vec3 wp, float t ) {
  float ph = dot( wp.xz, vec2( 0.043, 0.029 ) );
  float g = sin( t * 1.9 + ph ) * 0.5
          + sin( t * 0.97 + ph * 1.7 + 1.3 ) * 0.3
          + sin( t * 3.7 + ph * 2.3 ) * 0.2;
  float gust = 0.6 + 0.4 * sin( t * 0.23 + ph * 0.35 );
  return vec2( 0.82, 0.57 ) * ( g * gust );
}
`

/**
 * Drop-in replacement for `#include <project_vertex>`: applies the instance
 * transform, then displaces the world-space vertex by the wind field scaled
 * by the per-vertex `aSway` weight (with a slight droop so long sways arc
 * down instead of shearing sideways).
 */
const WIND_PROJECT_GLSL = /* glsl */ `
vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
  mvPosition = instanceMatrix * mvPosition;
#endif
{
  vec2 avW = avWind( mvPosition.xyz, uTime ) * aSway;
  mvPosition.xz += avW;
  mvPosition.y -= dot( avW, avW ) * 0.11;
}
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;
`

/**
 * Fragment patch for alpha-tested canvas foliage maps: canvas textures store
 * transparent texels as RGB 0, so mipmap averaging drags every distant sample
 * toward black — the classic "trees turn into dark blobs at range" bug.
 * Dividing by alpha recovers the mean color of the covering texels.
 */
const UNPREMULTIPLY_GLSL =
  'sampledDiffuseColor.rgb /= max( sampledDiffuseColor.a, 0.06 );\n\tdiffuseColor *= sampledDiffuseColor;'

/**
 * Sun-wrap + back-scatter translucency, appended after
 * `#include <emissivemap_fragment>` (so `diffuseColor` is final and the term
 * feeds `totalEmissiveRadiance`). Two parts, both scaled by sun elevation:
 * - a fixed ambient wrap that lifts shadow-facing canopy cards out of black
 *   (kills the "near-black broccoli puffball" read on the shade side), and
 * - a view-dependent back-scatter lobe so low sun glows THROUGH foliage when
 *   the camera looks sunward. Requires `vVegW` (world position varying) and
 *   the shared normalized `uSunDir` uniform.
 */
const TRANSLUCENCY_GLSL = /* glsl */ `
{
  float vegDay = clamp( uSunDir.y * 2.6, 0.0, 1.0 );
  vec3 vegV = normalize( vVegW - cameraPosition );
  float vegB = clamp( dot( vegV, uSunDir ), 0.0, 1.0 );
  totalEmissiveRadiance += diffuseColor.rgb * vegDay * ( pow( vegB, 3.0 ) * 0.55 + 0.16 );
}
`

/**
 * Installs the wind vertex patch on any material that renders tree geometry
 * (bark, foliage and the shadow-depth material), wiring the shared time
 * uniform so trees and their shadows move together. Pass `unpremultiply` for
 * materials whose color comes from an alpha-tested canvas map (see
 * {@link UNPREMULTIPLY_GLSL}). Pass `sun` (the shared normalized sun-direction
 * uniform) to add the {@link TRANSLUCENCY_GLSL} sun-wrap + back-scatter term —
 * foliage only; depth and bark materials must not receive it.
 */
function patchWind(
  mat: THREE.Material,
  uTime: { value: number },
  cacheKey: string,
  unpremultiply = false,
  sun?: { value: THREE.Vector3 },
): void {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uTime
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + WIND_GLSL)
      .replace('#include <project_vertex>', WIND_PROJECT_GLSL)
    if (sun) {
      shader.uniforms.uSunDir = sun
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vVegW;')
        .replace(
          'mvPosition = modelViewMatrix * mvPosition;',
          'vVegW = mvPosition.xyz;\nmvPosition = modelViewMatrix * mvPosition;',
        )
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vVegW;\nuniform vec3 uSunDir;')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' + TRANSLUCENCY_GLSL)
    }
    if (unpremultiply) {
      shader.fragmentShader = shader.fragmentShader.replace(
        'diffuseColor *= sampledDiffuseColor;',
        UNPREMULTIPLY_GLSL,
      )
    }
  }
  mat.customProgramCacheKey = () => cacheKey
}

// ---------------------------------------------------------------------------
// Canvas-painted textures (all procedural, no external assets)
// ---------------------------------------------------------------------------

/** Creates a canvas + 2D context of the given size. */
function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')
  if (!g) throw new Error('Vegetation: 2D canvas unavailable')
  return [c, g]
}

/** Wraps a canvas in an sRGB CanvasTexture with sensible filtering. */
function toTexture(canvas: HTMLCanvasElement, renderer: THREE.WebGLRenderer): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
  tex.generateMipmaps = true
  tex.minFilter = THREE.LinearMipmapLinearFilter
  return tex
}

/**
 * Foliage atlas, 512², four 256² tiles:
 * - tile (0,0): OPAQUE pale bark fill. Trunk vertices point here so the one
 *   shared alpha-tested shadow-depth material works for the whole tree.
 * - tiles (1,0), (0,1), (1,1): eucalypt leaf clusters — hundreds of narrow
 *   sickle leaves radiating from clump centers, olive→grey-green, with a few
 *   coppery young leaves. Transparent background, crisp under alphaTest.
 */
function paintFoliageAtlas(): HTMLCanvasElement {
  const [canvas, g] = makeCanvas(512, 512)
  const rng = mulberry32(0x1eaf5)

  // Bark tile: solid pale cream with faint vertical streaking.
  g.fillStyle = '#c9b8a6'
  g.fillRect(0, 0, 256, 256)
  for (let i = 0; i < 120; i++) {
    const x = rng() * 256
    const w = 2 + rng() * 10
    const shade = 175 + ((rng() * 60) | 0)
    g.fillStyle = `rgba(${shade},${shade - 12},${shade - 26},${0.25 + rng() * 0.3})`
    g.fillRect(x, 0, w, 256)
  }

  /** Paints one leaf-cluster tile at (ox, oy). */
  const tile = (ox: number, oy: number, seed: number): void => {
    const r = mulberry32(seed)
    // 3-4 clump centers per tile.
    const clumps = 3 + ((r() * 2) | 0)
    for (let c = 0; c < clumps; c++) {
      const cx = ox + 60 + r() * 136
      const cy = oy + 60 + r() * 136
      const leaves = 90 + ((r() * 50) | 0)
      for (let i = 0; i < leaves; i++) {
        const ang = r() * Math.PI * 2
        const dist = Math.pow(r(), 0.6) * 88
        const lx = cx + Math.cos(ang) * dist
        const ly = cy + Math.sin(ang) * dist
        const len = 9 + r() * 15
        const wid = 1.6 + r() * 2.2
        // Leaves hang: bias the leaf axis downward from the radial direction.
        const rot = lerp(ang, Math.PI / 2, 0.45) + (r() - 0.5) * 1.1
        const t = r()
        let cr: number, cg: number, cb: number
        if (t < 0.08) {
          cr = 148 + r() * 40 // coppery juvenile leaf
          cg = 82 + r() * 26
          cb = 48 + r() * 18
        } else {
          const s = r()
          cr = lerp(86, 128, s)
          cg = lerp(110, 132, s)
          cb = lerp(74, 104, s)
        }
        const edge = sat(1 - dist / 92)
        g.save()
        g.translate(lx, ly)
        g.rotate(rot)
        g.fillStyle = `rgba(${cr | 0},${cg | 0},${cb | 0},${0.55 + 0.45 * edge})`
        g.beginPath()
        g.ellipse(0, 0, len, wid, 0, 0, Math.PI * 2)
        g.fill()
        g.restore()
      }
    }
  }
  tile(256, 0, 0xa11e)
  tile(0, 256, 0xb22e)
  tile(256, 256, 0xc33e)
  return canvas
}

/**
 * Spinifex/tussock texture, 256²: dozens of thin arcing blades fanning out
 * from a root point at bottom-center, straw-gold with green shadows near the
 * base, seed-head dots at the tips. Transparent background for alphaTest.
 */
function paintGrassTexture(): HTMLCanvasElement {
  const [canvas, g] = makeCanvas(256, 256)
  const rng = mulberry32(0x9a55)
  const rootX = 128
  const rootY = 252
  for (let i = 0; i < 110; i++) {
    const lean = (rng() - 0.5) * 2.4
    const len = 110 + rng() * 130
    const tipX = rootX + Math.sin(lean) * len * (0.55 + rng() * 0.45)
    const tipY = rootY - Math.cos(lean * 0.55) * len
    const midX = rootX + (tipX - rootX) * 0.35 + (rng() - 0.5) * 14
    const midY = rootY - len * 0.55
    const t = rng()
    const cr = lerp(96, 196, t)
    const cg = lerp(104, 164, t)
    const cb = lerp(52, 92, t)
    g.strokeStyle = `rgba(${cr | 0},${cg | 0},${cb | 0},${0.75 + rng() * 0.25})`
    g.lineWidth = 1.2 + rng() * 2.4
    g.lineCap = 'round'
    g.beginPath()
    g.moveTo(rootX + (rng() - 0.5) * 26, rootY)
    g.quadraticCurveTo(midX, midY, tipX, tipY)
    g.stroke()
    if (rng() < 0.3) {
      g.fillStyle = `rgba(214,190,132,${0.5 + rng() * 0.4})`
      g.beginPath()
      g.ellipse(tipX, tipY, 2.4, 4.2, lean, 0, Math.PI * 2)
      g.fill()
    }
  }
  return canvas
}

/**
 * Saltbush/scrub texture, 256²: a low twiggy dome of hundreds of small
 * rounded grey-green leaves over dark stems, wider than tall, self-shaded
 * darker toward the base. Transparent background for alphaTest.
 */
function paintScrubTexture(): HTMLCanvasElement {
  const [canvas, g] = makeCanvas(256, 256)
  const rng = mulberry32(0x5c2b)
  // Twiggy stems fanning up from the root.
  for (let i = 0; i < 14; i++) {
    const lean = (rng() - 0.5) * 2.0
    const len = 90 + rng() * 90
    g.strokeStyle = `rgba(${60 + ((rng() * 30) | 0)},${52 + ((rng() * 24) | 0)},${40 + ((rng() * 18) | 0)},0.85)`
    g.lineWidth = 1.5 + rng() * 2
    g.lineCap = 'round'
    g.beginPath()
    g.moveTo(128 + (rng() - 0.5) * 30, 250)
    g.quadraticCurveTo(128 + lean * 40, 250 - len * 0.6, 128 + lean * 70, 250 - len)
    g.stroke()
  }
  // Leaf dome: dense center, ragged edge, darker near the ground.
  const domeCX = 128
  const domeCY = 148
  for (let i = 0; i < 760; i++) {
    const ang = rng() * Math.PI * 2
    const rr = Math.pow(rng(), 0.55)
    const lx = domeCX + Math.cos(ang) * rr * 112
    const ly = Math.min(244, domeCY + Math.sin(ang) * rr * 82)
    // Vertical self-shade: bright crown, dusky underside.
    const shade = 1.18 - sat((ly - 60) / 190) * 0.55
    const t = rng()
    const cr = lerp(96, 142, t) * shade
    const cg = lerp(112, 150, t) * shade
    const cb = lerp(78, 108, t) * shade
    g.fillStyle = `rgba(${cr | 0},${cg | 0},${cb | 0},${0.6 + rng() * 0.4})`
    g.save()
    g.translate(lx, ly)
    g.rotate(rng() * Math.PI)
    g.beginPath()
    g.ellipse(0, 0, 2.2 + rng() * 3.4, 1.6 + rng() * 2.2, 0, 0, Math.PI * 2)
    g.fill()
    g.restore()
  }
  // A few pale seed/straw flecks catch the light at the crown.
  for (let i = 0; i < 60; i++) {
    const ang = rng() * Math.PI * 2
    const rr = Math.pow(rng(), 0.5)
    const lx = domeCX + Math.cos(ang) * rr * 100
    const ly = domeCY - 20 + Math.sin(ang) * rr * 55
    g.fillStyle = `rgba(206,192,148,${0.35 + rng() * 0.35})`
    g.beginPath()
    g.arc(lx, ly, 1 + rng() * 1.6, 0, Math.PI * 2)
    g.fill()
  }
  return canvas
}

// ---------------------------------------------------------------------------
// Tree archetypes
// ---------------------------------------------------------------------------

/** Growth + palette parameters for one gum-tree archetype. */
interface TreeSpec {
  /** Debug name. */
  name: string
  /** Trunk height range, meters. */
  trunkH: [number, number]
  /** Trunk base radius as a fraction of trunk height. */
  trunkR: number
  /** Recursive branch levels beyond the trunk. */
  levels: number
  /** Children spawned per branch tip. */
  kids: [number, number]
  /** Branch length as a fraction of the parent's. */
  lenRatio: number
  /** Branch spread half-angle, radians. */
  spread: number
  /** Pull of every branch back toward vertical (gums reach for the sun). */
  upBias: number
  /** Bark gradient: base color → upper-limb color. */
  barkA: THREE.Color
  barkB: THREE.Color
  /** Canopy card tint range. */
  leafA: THREE.Color
  leafB: THREE.Color
  /** Canopy card half-size, meters (pre instance scale). */
  cardSize: number
  /** Leaf-cluster cards per branch tip. */
  cardsPerTip: [number, number]
}

/** The four archetypes, indexed 0..3. */
function makeSpecs(): TreeSpec[] {
  return [
    {
      name: 'river-red-gum',
      trunkH: [7, 10],
      trunkR: 0.062,
      levels: 3,
      kids: [2, 3],
      lenRatio: 0.72,
      spread: 0.85,
      upBias: 0.16,
      barkA: new THREE.Color(0.62, 0.56, 0.5),
      barkB: new THREE.Color(0.82, 0.76, 0.68),
      leafA: new THREE.Color(0.38, 0.46, 0.3),
      leafB: new THREE.Color(0.52, 0.56, 0.4),
      cardSize: 2.7,
      cardsPerTip: [3, 5],
    },
    {
      name: 'ghost-gum',
      trunkH: [5.5, 8],
      trunkR: 0.05,
      levels: 3,
      kids: [2, 3],
      lenRatio: 0.68,
      spread: 0.62,
      upBias: 0.3,
      barkA: new THREE.Color(0.88, 0.85, 0.8),
      barkB: new THREE.Color(0.97, 0.95, 0.9),
      leafA: new THREE.Color(0.42, 0.48, 0.32),
      leafB: new THREE.Color(0.56, 0.58, 0.42),
      cardSize: 2.0,
      cardsPerTip: [2, 4],
    },
    {
      name: 'stringybark',
      trunkH: [10, 14],
      trunkR: 0.055,
      levels: 3,
      kids: [2, 3],
      lenRatio: 0.62,
      spread: 0.55,
      upBias: 0.34,
      barkA: new THREE.Color(0.38, 0.3, 0.24),
      barkB: new THREE.Color(0.56, 0.47, 0.38),
      leafA: new THREE.Color(0.3, 0.4, 0.26),
      leafB: new THREE.Color(0.44, 0.5, 0.34),
      cardSize: 2.3,
      cardsPerTip: [3, 4],
    },
    {
      name: 'mulga',
      trunkH: [2.2, 3.4],
      trunkR: 0.075,
      levels: 2,
      kids: [3, 4],
      lenRatio: 0.78,
      spread: 1.0,
      upBias: 0.1,
      barkA: new THREE.Color(0.32, 0.27, 0.22),
      barkB: new THREE.Color(0.48, 0.42, 0.34),
      leafA: new THREE.Color(0.42, 0.45, 0.3),
      leafB: new THREE.Color(0.56, 0.55, 0.36),
      cardSize: 1.35,
      cardsPerTip: [2, 3],
    },
  ]
}

/**
 * Impostor silhouette atlas: 2048×512, eight 256×512 tiles — TWO silhouette
 * variants for each of the four archetypes (column = archetype*2 + variant).
 * Every tile bakes a directional sun gradient: warm-lit canopy on the tile's
 * LEFT and top, cool blue-grey shade on the right and underside. The far-tree
 * shader mirrors the card whenever the sun sits on the other side, so distant
 * trees always show a proper lit side and a shadow side instead of one flat
 * dark blob. Alpha is eroded with speckle so silhouettes read ragged.
 */
function paintImpostorAtlas(specs: TreeSpec[]): HTMLCanvasElement {
  const [canvas, g] = makeCanvas(2048, 512)
  const base = new THREE.Color()
  const lit = new THREE.Color()
  const shade = new THREE.Color()
  for (let a = 0; a < specs.length; a++) {
    const s = specs[a]
    for (let v = 0; v < 2; v++) {
      const rng = mulberry32(0x1417 + a * 977 + v * 5501)
      const ox = (a * 2 + v) * 256
      // Clip to this tile: canopy blobs must never bleed into the neighbour
      // archetype's tile, or cards grow alien color slivers at their edges.
      g.save()
      g.beginPath()
      g.rect(ox, 0, 256, 512)
      g.clip()
      const cx = ox + 128
      const baseY = 500
      const trunkTop = (a === 3 ? 330 : 210) + (v === 1 ? 28 : 0) // mulga is squat; variant 1 sits lower
      // Trunk + limbs (variant 1 forks differently).
      base.copy(s.barkB)
      for (let i = 0; i < 3 + v; i++) {
        const spreadX = (i - 1 - v * 0.5) * 26 + (rng() - 0.5) * 20
        const w = i === 1 ? 13 : 7
        g.strokeStyle = `rgb(${(base.r * 235) | 0},${(base.g * 235) | 0},${(base.b * 235) | 0})`
        g.lineWidth = w
        g.lineCap = 'round'
        g.beginPath()
        g.moveTo(cx + spreadX * 0.2, baseY)
        g.quadraticCurveTo(cx + spreadX * 0.7, (baseY + trunkTop) / 2, cx + spreadX * 1.6, trunkTop + 40)
        g.stroke()
      }
      // Canopy: layered soft blobs with the baked lateral+vertical sun gradient.
      const blobs = 30 + ((rng() * 12) | 0)
      const spanX = (a === 0 ? 108 : a === 3 ? 96 : 80) * (v === 1 ? 0.86 : 1)
      const spanY = (a === 3 ? 70 : 105) * (v === 1 ? 1.12 : 1)
      for (let i = 0; i < blobs; i++) {
        const bx = cx + (rng() - 0.5) * 2 * spanX
        const by = trunkTop - 20 + (rng() - 0.5) * 2 * spanY
        const r = 15 + rng() * 30
        const latT = sat(0.5 - (bx - cx) / (2.4 * spanX)) // 1 at the lit (left) edge
        const vertT = sat(1 - (by - (trunkTop - 20 - spanY)) / (2 * spanY)) // 1 at the crown
        const litT = sat(latT * 0.62 + vertT * 0.55)
        base.copy(s.leafA).lerp(s.leafB, rng())
        lit.setRGB(Math.min(1, base.r * 1.6 + 0.1), Math.min(1, base.g * 1.5 + 0.08), Math.min(1, base.b * 1.15))
        shade.setRGB(base.r * 0.42, base.g * 0.46, base.b * 0.6)
        base.copy(shade).lerp(lit, litT)
        const grad = g.createRadialGradient(bx, by, r * 0.1, bx, by, r)
        grad.addColorStop(0, `rgba(${(base.r * 255) | 0},${(base.g * 255) | 0},${(base.b * 255) | 0},0.9)`)
        grad.addColorStop(1, `rgba(${(base.r * 220) | 0},${(base.g * 220) | 0},${(base.b * 220) | 0},0)`)
        g.fillStyle = grad
        g.beginPath()
        g.arc(bx, by, r, 0, Math.PI * 2)
        g.fill()
      }
      // Erode the canopy so the silhouette is ragged.
      g.globalCompositeOperation = 'destination-out'
      for (let i = 0; i < 240; i++) {
        const bx = ox + rng() * 256
        const by = rng() * 460
        g.fillStyle = `rgba(0,0,0,${0.25 + rng() * 0.5})`
        g.beginPath()
        g.arc(bx, by, 1.5 + rng() * 5, 0, Math.PI * 2)
        g.fill()
      }
      g.globalCompositeOperation = 'source-over'
      g.restore()
    }
  }
  return canvas
}

// ---------------------------------------------------------------------------
// Tree geometry grower
// ---------------------------------------------------------------------------

/** A grown archetype: merged bark+canopy geometry plus placement metrics. */
interface TreeArchetype {
  geometry: THREE.BufferGeometry
  /** Overall height, meters (for impostor card sizing). */
  height: number
  /** Canopy half-width, meters. */
  radius: number
  spec: TreeSpec
}

/**
 * Grows one archetype into a single indexed BufferGeometry with two groups:
 * group 0 = trunk/branches (bark material), group 1 = canopy cards (foliage
 * material). Attributes: position, normal, uv, color (baked bark streaks /
 * leaf tints) and aSway (wind weight, 0 at the roots → ~1 at the leaf tips).
 */
function growTree(spec: TreeSpec, seed: number): TreeArchetype {
  const rng = mulberry32(seed)
  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const colors: number[] = []
  const sways: number[] = []
  const indices: number[] = []
  const tips: { x: number; y: number; z: number; sway: number }[] = []
  let maxY = 0
  let maxR = 0

  const col = new THREE.Color()
  const dir = new THREE.Vector3()
  const side = new THREE.Vector3()
  const binorm = new THREE.Vector3()
  const point = new THREE.Vector3()
  const childDir = new THREE.Vector3()
  const axis = new THREE.Vector3()
  const quat = new THREE.Quaternion()

  const trunkH = lerp(spec.trunkH[0], spec.trunkH[1], rng())
  const RADIAL = [7, 5, 4, 3] as const

  /**
   * Emits one tapered, slightly curved tube and recurses into children.
   * Rings connect start→end; each ring carries bark color + sway weight.
   */
  const branch = (
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    len: number,
    rad: number,
    level: number,
    sway0: number,
  ): void => {
    dir.set(dx, dy, dz).normalize()
    const segs = level === 0 ? 3 : 2
    const radial = RADIAL[Math.min(level, RADIAL.length - 1)]
    const swayGain = 0.1 + 0.16 * level
    let px = ox
    let py = oy
    let pz = oz
    const ringStart: number[] = []
    for (let si = 0; si <= segs; si++) {
      const t = si / segs
      // Drift the growth direction: wander + up bias, stronger along the branch.
      if (si > 0) {
        dir.x += (rng() - 0.5) * 0.34
        dir.z += (rng() - 0.5) * 0.34
        dir.y += spec.upBias * 0.7
        dir.normalize()
        const step = len / segs
        px += dir.x * step
        py += dir.y * step
        pz += dir.z * step
      }
      // Orthonormal frame around dir.
      side.set(0, 1, 0)
      if (Math.abs(dir.y) > 0.94) side.set(1, 0, 0)
      side.crossVectors(side, dir).normalize()
      binorm.crossVectors(dir, side)
      const r = rad * (1 - 0.62 * t)
      const swayHere = sway0 + swayGain * t
      // Bark color: gradient up the tree + per-ring streak jitter.
      const hNorm = sat(py / (trunkH * 2.2))
      col.copy(spec.barkA).lerp(spec.barkB, sat(hNorm + (rng() - 0.5) * 0.3))
      const mul = 0.9 + rng() * 0.2
      ringStart.push(positions.length / 3)
      for (let k = 0; k < radial; k++) {
        const a = (k / radial) * Math.PI * 2
        const ca = Math.cos(a)
        const sa = Math.sin(a)
        point.set(
          px + (side.x * ca + binorm.x * sa) * r,
          py + (side.y * ca + binorm.y * sa) * r,
          pz + (side.z * ca + binorm.z * sa) * r,
        )
        positions.push(point.x, point.y, point.z)
        normals.push(side.x * ca + binorm.x * sa, side.y * ca + binorm.y * sa, side.z * ca + binorm.z * sa)
        // Opaque bark tile: canvas top-left = UV (0..0.5, 0.5..1) after flipY.
        uvs.push(0.06 + 0.13 * (k / radial), 0.56 + 0.13 * t)
        colors.push(col.r * mul, col.g * mul, col.b * mul)
        sways.push(swayHere)
      }
      if (py > maxY) maxY = py
    }
    // Stitch consecutive rings.
    for (let si = 0; si < segs; si++) {
      const r0 = ringStart[si]
      const r1 = ringStart[si + 1]
      for (let k = 0; k < radial; k++) {
        const k1 = (k + 1) % radial
        indices.push(r0 + k, r0 + k1, r1 + k, r0 + k1, r1 + k1, r1 + k)
      }
    }
    const tipSway = sway0 + swayGain
    const hr = Math.hypot(px, pz)
    if (hr > maxR) maxR = hr
    if (level >= spec.levels) {
      tips.push({ x: px, y: py, z: pz, sway: tipSway })
      return
    }
    // Children fan out around the parent direction.
    const n = spec.kids[0] + Math.round(rng() * (spec.kids[1] - spec.kids[0]))
    const azimuth0 = rng() * Math.PI * 2
    for (let c = 0; c < n; c++) {
      const az = azimuth0 + (c / n) * Math.PI * 2 + (rng() - 0.5) * 0.8
      const tilt = spec.spread * (0.55 + rng() * 0.6)
      axis.set(Math.cos(az), 0, Math.sin(az))
      // Tilt dir away from itself around a horizontal axis ⊥ to azimuth.
      axis.crossVectors(dir, axis)
      if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0)
      axis.normalize()
      quat.setFromAxisAngle(axis, tilt)
      childDir.copy(dir).applyQuaternion(quat)
      childDir.y += spec.upBias
      childDir.normalize()
      // Some children start partway along the parent, not all at the tip.
      const bt = level === 0 ? 0.55 + rng() * 0.45 : 0.75 + rng() * 0.25
      branch(
        ox + (px - ox) * bt,
        oy + (py - oy) * bt,
        oz + (pz - oz) * bt,
        childDir.x,
        childDir.y,
        childDir.z,
        len * spec.lenRatio * (0.8 + rng() * 0.4),
        rad * (1 - 0.62 * bt) * 0.72,
        level + 1,
        sway0 + swayGain * bt,
      )
    }
    // Ragged extra tip on the parent line keeps crowns from looking combed.
    tips.push({ x: px, y: py, z: pz, sway: tipSway })
  }

  branch(0, 0, 0, (rng() - 0.5) * 0.14, 1, (rng() - 0.5) * 0.14, trunkH, trunkH * spec.trunkR, 0, 0)
  const trunkIndexCount = indices.length

  // --- Canopy cards at every branch tip ---
  // Leaf-cluster tile UV origins (canvas coordinates flipped vertically).
  const LEAF_TILES: [number, number][] = [
    [0.5, 0.5],
    [0.0, 0.0],
    [0.5, 0.0],
  ]
  const canopyCenterY = maxY * 0.78
  for (const tip of tips) {
    const cards = spec.cardsPerTip[0] + Math.round(rng() * (spec.cardsPerTip[1] - spec.cardsPerTip[0]))
    for (let c = 0; c < cards; c++) {
      const s = spec.cardSize * (0.75 + rng() * 0.6)
      const cx = tip.x + (rng() - 0.5) * s * 1.4
      const cy = tip.y + (rng() - 0.3) * s * 1.1
      const cz = tip.z + (rng() - 0.5) * s * 1.4
      quat.setFromEuler(new THREE.Euler(rng() * Math.PI, rng() * Math.PI * 2, rng() * Math.PI, 'YXZ'))
      side.set(1, 0, 0).applyQuaternion(quat).multiplyScalar(s)
      binorm.set(0, 1, 0).applyQuaternion(quat).multiplyScalar(s)
      // Blend the face normal toward "up + outward" for soft canopy lighting.
      dir.set(0, 0, 1).applyQuaternion(quat)
      point.set(cx, cy - canopyCenterY, cz).normalize()
      dir.multiplyScalar(0.35).addScaledVector(point, 0.45).y += 0.75
      dir.normalize()
      // Lower/inner canopy sits in shade — bake it darker.
      const shade = lerp(0.55, 1.12, sat((cy - canopyCenterY * 0.35) / (maxY - canopyCenterY * 0.35 + 0.001)))
      col.copy(spec.leafA).lerp(spec.leafB, rng()).multiplyScalar(shade)
      const [tu, tv] = LEAF_TILES[(rng() * 3) | 0]
      const base = positions.length / 3
      const sway = tip.sway + 0.12 + rng() * 0.14
      for (let v = 0; v < 4; v++) {
        const sx = v === 1 || v === 2 ? 1 : -1
        const sy = v >= 2 ? 1 : -1
        positions.push(cx + side.x * sx + binorm.x * sy, cy + side.y * sx + binorm.y * sy, cz + side.z * sx + binorm.z * sy)
        normals.push(dir.x, dir.y, dir.z)
        uvs.push(tu + (sx > 0 ? 0.485 : 0.015), tv + (sy > 0 ? 0.485 : 0.015))
        colors.push(col.r, col.g, col.b)
        sways.push(sway)
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
      const cr = Math.hypot(cx, cz) + s
      if (cr > maxR) maxR = cr
      if (cy + s > maxY) maxY = cy + s
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  geo.setAttribute('aSway', new THREE.Float32BufferAttribute(sways, 1))
  geo.setIndex(indices)
  geo.addGroup(0, trunkIndexCount, 0)
  geo.addGroup(trunkIndexCount, indices.length - trunkIndexCount, 1)
  geo.computeBoundingSphere()
  return { geometry: geo, height: maxY, radius: maxR, spec }
}

// ---------------------------------------------------------------------------
// Granite boulders
// ---------------------------------------------------------------------------

/**
 * One boulder variant: an icosphere displaced by two octaves of 3-D simplex
 * noise, squashed vertically so it sits like weathered granite rather than a
 * floating ball. Unit-ish radius; instances scale it 1–7 m.
 */
function makeBoulderGeometry(seed: number, squash: number): THREE.BufferGeometry {
  const noise3 = createNoise3D(mulberry32(seed))
  const geo = new THREE.IcosahedronGeometry(1, 2)
  const pos = geo.getAttribute('position') as THREE.BufferAttribute
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i)
    const n = 1 + 0.3 * noise3(v.x * 1.15, v.y * 1.15, v.z * 1.15) + 0.09 * noise3(v.x * 3.7, v.y * 3.7, v.z * 3.7)
    v.multiplyScalar(n)
    v.y *= squash
    pos.setXYZ(i, v.x, v.y, v.z)
  }
  geo.computeVertexNormals()
  geo.computeBoundingSphere()
  return geo
}

/**
 * Granite material: mid-grey base, fragment-shader mineral speckle, and
 * warm lichen staining on upward faces — all keyed off world position so
 * every instance weathers differently with zero textures.
 */
function makeGraniteMaterial(): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0.0 })
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vRockPos;\nvarying vec3 vRockN;')
      .replace(
        '#include <project_vertex>',
        '#include <project_vertex>\n  vRockPos = ( modelMatrix * ( instanceMatrix * vec4( transformed, 1.0 ) ) ).xyz;\n  vRockN = normalize( mat3( modelMatrix ) * mat3( instanceMatrix ) * objectNormal );',
      )
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vRockPos;\nvarying vec3 vRockN;\n' +
          'float rkHash( vec3 p ) {\n' +
          '  p = fract( p * 0.3183099 + 0.1 );\n' +
          '  p *= 17.0;\n' +
          '  return fract( p.x * p.y * p.z * ( p.x + p.y + p.z ) );\n' +
          '}',
      )
      .replace(
        '#include <color_fragment>',
        '#include <color_fragment>\n{\n' +
          '  vec3 base = vec3( 0.30, 0.275, 0.25 );\n' +
          '  float macro = rkHash( floor( vRockPos * 0.6 ) );\n' +
          '  base *= 0.8 + 0.5 * macro;\n' +
          '  float sp = rkHash( floor( vRockPos * 14.0 ) );\n' +
          '  base = mix( base, vec3( 0.10, 0.095, 0.09 ), smoothstep( 0.78, 0.9, sp ) );\n' +
          '  base = mix( base, vec3( 0.55, 0.52, 0.48 ), smoothstep( 0.9, 0.985, sp ) );\n' +
          '  float lich = smoothstep( 0.55, 0.85, rkHash( floor( vRockPos * 1.7 + 31.0 ) ) )\n' +
          '             * smoothstep( 0.15, 0.55, vRockN.y );\n' +
          '  base = mix( base, vec3( 0.42, 0.20, 0.07 ), lich * 0.55 );\n' +
          '  diffuseColor.rgb *= base;\n' +
          '}',
      )
  }
  mat.customProgramCacheKey = () => 'aviary-granite-v1'
  return mat
}

// ---------------------------------------------------------------------------
// Grass material
// ---------------------------------------------------------------------------

/**
 * Ground-cover material (spinifex tussocks AND saltbush scrub): alpha-tested
 * crossed cards, wind-bent at the blade tips and scaled to zero at the ring
 * edge so density fades with distance instead of popping. A second scale-fade
 * runs over the `altFade0`..`altFade1` camera-altitude band (shared `uAlt`
 * uniform), so climbing out of ground-cover range shrinks every instance to
 * nothing instead of popping the whole layer off at the cull altitude. Gets
 * the same sun-wrap + back-scatter term as the tree foliage (shared `sun`
 * uniform) so tussocks glow rim-lit at low sun instead of going black.
 * Per-instance tint arrives via instanceColor. `radius` must match the
 * scatter ring radius.
 */
function makeGroundCoverMaterial(
  map: THREE.Texture,
  uTime: { value: number },
  radius: number,
  altFade0: number,
  altFade1: number,
  sun: { value: THREE.Vector3 },
  uAlt: { value: number },
  cacheKey: string,
): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({
    map,
    alphaTest: 0.38,
    side: THREE.DoubleSide,
    roughness: 0.9,
    metalness: 0.0,
  })
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uTime
    shader.uniforms.uSunDir = sun
    shader.uniforms.uAlt = uAlt
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vVegW;\nuniform vec3 uSunDir;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' + TRANSLUCENCY_GLSL)
      .replace('diffuseColor *= sampledDiffuseColor;', UNPREMULTIPLY_GLSL)
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vVegW;\nuniform float uAlt;\n' +
          WIND_GLSL.replace('attribute float aSway;', ''),
      )
      .replace(
        '#include <project_vertex>',
        /* glsl */ `
vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
  // Shrink toward the root as the instance nears the ring edge, and shrink
  // the whole layer out across the altitude fade band.
  float gd = distance( instanceMatrix[ 3 ].xz, cameraPosition.xz );
  float gFade = ( 1.0 - smoothstep( ${(radius * 0.62).toFixed(1)}, ${radius.toFixed(1)}, gd ) )
              * ( 1.0 - smoothstep( ${altFade0.toFixed(1)}, ${altFade1.toFixed(1)}, uAlt ) );
  mvPosition.xyz *= gFade;
  mvPosition = instanceMatrix * mvPosition;
#endif
{
  float bend = transformed.y * transformed.y * 0.55;
  vec2 avW = avWind( mvPosition.xyz, uTime ) * bend;
  mvPosition.xz += avW;
}
vVegW = mvPosition.xyz;
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;
`,
      )
  }
  mat.customProgramCacheKey = () => cacheKey
  return mat
}

/** Crossed-card ground-cover geometry: `planes` W×H quads at even yaw steps. */
function makeCrossCardGeometry(W: number, H: number, planes: number): THREE.BufferGeometry {
  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []
  for (let p = 0; p < planes; p++) {
    const a = (p / planes) * Math.PI
    const ca = Math.cos(a)
    const sa = Math.sin(a)
    const base = positions.length / 3
    for (let v = 0; v < 4; v++) {
      const sx = v === 1 || v === 2 ? W : -W
      const y = v >= 2 ? H : 0
      positions.push(ca * sx, y, sa * sx)
      normals.push(-sa, 0.25, ca)
      uvs.push(sx > 0 ? 1 : 0, y > 0 ? 1 : 0)
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geo.setIndex(indices)
  return geo
}

// ---------------------------------------------------------------------------
// Impostor shader
// ---------------------------------------------------------------------------

/**
 * Impostor card geometry: four unit quads sharing one instance stream.
 * Quad 0 becomes the far cylindrical billboard; quads 1 and 2 become a
 * fixed world-space cross for the mid range; quad 3 becomes a horizontal
 * overhead canopy card that fades in as the camera looks down. The vertex
 * shader collapses whichever set is inactive to zero width (degenerate
 * triangles — no fill cost).
 */
function makeImpostorGeometry(): THREE.InstancedBufferGeometry {
  const positions: number[] = []
  const uvs: number[] = []
  const planes: number[] = []
  const indices: number[] = []
  for (let p = 0; p < 4; p++) {
    const base = p * 4
    positions.push(-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0)
    uvs.push(0, 0, 1, 0, 1, 1, 0, 1)
    planes.push(p, p, p, p)
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
  }
  const geo = new THREE.InstancedBufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geo.setAttribute('aPlane', new THREE.Float32BufferAttribute(planes, 1))
  geo.setIndex(indices)
  return geo
}

/**
 * Builds the far-tree impostor material. Cards fade in exactly where the CPU
 * hides the full-geometry chunk (both compare horizontal distance to the same
 * 750 m chunk center, dithered). Between {@link TREE_NEAR} and
 * {@link IMPOSTOR_CROSS} each tree renders as two fixed world-space cross
 * cards (world-anchored, so aerial parallax reads as volume); beyond that a
 * single cylindrical billboard takes over. Every card samples the sun-shaded
 * 8-tile atlas and mirrors its UVs so the baked warm-lit side faces the real
 * sun and the cool shade side faces away.
 */
function makeImpostorMaterial(map: THREE.Texture, uAltBoost: { value: number }): THREE.ShaderMaterial {
  const mat = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      {
        uMap: { value: map },
        uSun: { value: new THREE.Vector3(0.3, 0.6, -0.5) },
      },
    ]),
    vertexShader: /* glsl */ `
      #include <common>
      #include <fog_pars_vertex>
      attribute vec3 iPos;
      attribute vec2 iDim;
      attribute vec3 iTint;
      attribute float iCol;
      attribute float aPlane;
      varying vec2 vUv;
      varying vec3 vTint;
      varying float vFade;
      varying float vLight;
      varying float vTop;
      uniform vec3 uSun;
      uniform float uAltBoost;
      void main() {
        // Fade in exactly where the CPU hides the full-geometry chunk.
        // uAltBoost inflates the LOD distance as the camera climbs (same
        // scalar the CPU test uses), so aerial views hand alpha-tested leaf
        // cards over to impostor crowns instead of degrading into speckle.
        vec2 cc = ( floor( iPos.xz / ${CHUNK_SIZE.toFixed(1)} ) + 0.5 ) * ${CHUNK_SIZE.toFixed(1)};
        float cd = distance( cameraPosition.xz, cc ) + uAltBoost;
        vFade = smoothstep( ${(TREE_NEAR - IMPOSTOR_FADE).toFixed(1)}, ${TREE_NEAR.toFixed(1)}, cd );
        vec3 toCam = cameraPosition - iPos;
        // LOD blend: fixed cross cards mid-range, one billboard far. The
        // inactive set collapses to zero width — degenerate, no fill cost.
        float farB = smoothstep( ${IMPOSTOR_CROSS.toFixed(1)}, ${(IMPOSTOR_CROSS + IMPOSTOR_CROSS_FADE).toFixed(1)}, cd );
        // Overhead blend: vertical cards go edge-on from altitude, so a
        // horizontal canopy card fades in as the view angle steepens.
        float dn = clamp( toCam.y / max( length( toCam ), 1e-3 ), 0.0, 1.0 );
        float topB = smoothstep( 0.5, 0.78, dn );
        vec2 sunH = normalize( uSun.xz + vec2( 1e-4 ) );
        float day = clamp( uSun.y * 2.2, 0.0, 1.0 );
        vec3 wp;
        if ( aPlane > 2.5 ) {
          // Horizontal crown card at canopy height, sized by canopy width.
          wp = iPos + vec3( position.x, 0.0, position.y ) * ( iDim.x * 1.08 * topB )
             + vec3( 0.0, iDim.y * 0.66, 0.0 );
          // Sample the canopy band of the silhouette tile — a blobby crown.
          vUv = vec2( ( uv.x + iCol ) * 0.125, 0.40 + uv.y * 0.42 );
          vTop = 1.0;
          // Crowns seen from above are mostly sunlit.
          vLight = 0.62 + 0.5 * day;
        } else {
          vec3 axis;
          float w = position.x;
          if ( aPlane < 0.5 ) {
            // Cylindrical billboard around the tree's vertical axis.
            axis = normalize( vec3( toCam.z, 0.0, -toCam.x ) + vec3( 1e-4, 0.0, 0.0 ) );
            w *= farB;
          } else {
            // World-anchored cross, azimuth hashed per tree.
            float ya = fract( sin( dot( iPos.xz, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ) * 3.14159;
            if ( aPlane > 1.5 ) ya += 1.5708;
            axis = vec3( cos( ya ), 0.0, sin( ya ) );
            w *= 1.0 - farB;
          }
          // The painted lit side is the tile's left (u = 0), rendered at -axis.
          // Mirror the card whenever the sun sits on the +axis side instead, so
          // the baked lit/shadow sides always face the real sun.
          float flip = step( 0.0, dot( sunH, axis.xz ) );
          float u = mix( uv.x, 1.0 - uv.x, flip );
          vUv = vec2( ( u + iCol ) * 0.125, uv.y );
          vTop = 0.0;
          // Aggregate shading: sky ambient by sun elevation, plus a boost when
          // the camera looks down-sun (seeing mostly lit canopy) and a dip when
          // it looks into the shadow sides.
          float face = dot( normalize( toCam.xz + vec2( 1e-4 ) ), sunH );
          vLight = ( 0.52 + 0.48 * day ) * ( 1.0 + 0.16 * face );
          wp = iPos + axis * w * iDim.x + vec3( 0.0, ( position.y + 0.5 ) * iDim.y, 0.0 );
        }
        vTint = iTint;
        vec4 mvPosition = viewMatrix * vec4( wp, 1.0 );
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      #include <common>
      #include <fog_pars_fragment>
      uniform sampler2D uMap;
      varying vec2 vUv;
      varying vec3 vTint;
      varying float vFade;
      varying float vLight;
      varying float vTop;
      void main() {
        vec4 tex = texture2D( uMap, vUv );
        // Solid coverage cut: steady-state far trees are a crisp silhouette,
        // never a stochastic sprinkle.
        if ( tex.a < 0.30 ) discard;
        // Dithered cross-fade ONLY inside the near/far hand-over band —
        // vFade is 1 everywhere beyond it, so no distant speckle.
        float dither = fract( dot( gl_FragCoord.xy, vec2( 0.7548776, 0.5698402 ) ) );
        if ( vFade < dither ) discard;
        // Canvas mips average transparent (RGB 0) texels into the color —
        // divide by alpha to recover the true foliage color at range.
        vec3 col = ( tex.rgb / max( tex.a, 0.06 ) ) * vTint * vLight;
        // Vertical AO gradient reinforces the baked one: lit crown on top,
        // dark shaded underside — kills the flat "broccoli puffball" read.
        col *= mix( 0.66 + 0.50 * vUv.y, 1.0, vTop );
        gl_FragColor = vec4( col, 1.0 );
        #include <fog_fragment>
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
    fog: true,
  })
  // Merge clones uniform entries — assign the shared LOD-boost object AFTER
  // construction so the CPU chunk walk and this shader read one scalar.
  mat.uniforms.uAltBoost = uAltBoost
  return mat
}

/**
 * Soft blob-shadow decal material. One instanced quad per tree (sharing the
 * impostor's iPos/iDim buffers, plus a baked per-tree ground normal) hugs
 * the terrain's tangent plane and darkens it under the canopy, stretched
 * away from the live sun azimuth. Real shadow maps only reach the
 * {@link TREE_SHADOW} chunk ring, so the blob ramps to full strength beyond
 * it — grounding every tree to the horizon — while keeping a light
 * ambient-occlusion pool under near trees. Transparent, no depth write.
 */
function makeBlobShadowMaterial(uAltBoost: { value: number }): THREE.ShaderMaterial {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uSun: { value: new THREE.Vector3(0.3, 0.6, -0.5) },
      uAltBoost,
    },
    vertexShader: /* glsl */ `
      attribute vec3 iPos;
      attribute vec2 iDim;
      attribute vec3 iNorm;
      uniform vec3 uSun;
      uniform float uAltBoost;
      varying vec2 vLocal;
      varying float vA;
      void main() {
        // Same altitude-boosted chunk-center test as the tree LODs: blobs
        // strengthen exactly where real shadow-map casting stops.
        vec2 cc = ( floor( iPos.xz / ${CHUNK_SIZE.toFixed(1)} ) + 0.5 ) * ${CHUNK_SIZE.toFixed(1)};
        float cd = distance( cameraPosition.xz, cc ) + uAltBoost;
        float nearB = smoothstep( ${(TREE_SHADOW - 260).toFixed(1)}, ${TREE_SHADOW.toFixed(1)}, cd );
        float r = iDim.x * 0.58;
        vec2 sunH = normalize( uSun.xz + vec2( 1e-4 ) );
        // Low sun stretches the ellipse away from the sun azimuth.
        float e = 1.0 + clamp( 0.55 * ( 1.0 - uSun.y ) / max( uSun.y, 0.22 ), 0.0, 1.7 );
        vec2 major = -sunH;
        vec2 minor = vec2( -major.y, major.x );
        vLocal = position.xz * 2.0;
        vec2 off = major * ( vLocal.x * r * e + r * ( e - 1.0 ) * 0.55 )
                 + minor * ( vLocal.y * r );
        vec3 wp = iPos + vec3( off.x, 0.0, off.y );
        // Hug the terrain's tangent plane (per-tree ground normal), then
        // lift slightly — more with distance — to stay clear of z-fighting.
        wp.y = iPos.y - ( iNorm.x * off.x + iNorm.z * off.y ) / max( iNorm.y, 0.55 );
        float dCam = distance( cameraPosition, wp );
        wp.y += 0.35 + dCam * 0.001;
        // Strength: AO pool near, full shadow beyond the shadow-map ring,
        // scaled by sun elevation and faded out toward the fog line.
        vA = ( 0.34 + 0.30 * nearB )
           * clamp( uSun.y * 3.0, 0.0, 1.0 )
           * ( 1.0 - smoothstep( 3800.0, 6500.0, dCam ) );
        gl_Position = projectionMatrix * viewMatrix * vec4( wp, 1.0 );
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vLocal;
      varying float vA;
      void main() {
        float d = length( vLocal );
        float a = 1.0 - smoothstep( 0.35, 1.0, d );
        gl_FragColor = vec4( vec3( 0.035, 0.045, 0.04 ), a * a * vA );
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
  return mat
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/** Per-chunk bookkeeping for the near/far and shadow hand-overs. */
interface VegChunk {
  /** Chunk center, world XZ. */
  cx: number
  cz: number
  /** Full-geometry tree meshes living in this chunk (one per archetype). */
  meshes: THREE.InstancedMesh[]
  visible: boolean
  shadows: boolean
}

/**
 * The Vegetation game module. Owns the gum forest (near instanced geometry +
 * far impostors), the wind-blown spinifex ring and the granite boulders.
 */
export class Vegetation implements GameModule {
  readonly name = 'vegetation'

  /** Shared wind clock for every vegetation shader. */
  private readonly uTime = { value: 0 }
  /** Shared normalized sun direction for the foliage translucency term. */
  private readonly uSunDir = { value: new THREE.Vector3(0.35, 0.4, -0.55).normalize() }
  /** Shared LOD-distance boost from camera altitude (see refreshChunks). */
  private readonly uAltBoost = { value: 0 }
  /** Shared camera altitude over terrain, meters (ground-cover fade). */
  private readonly uAlt = { value: 0 }
  private chunks: VegChunk[] = []
  private impostorMat!: THREE.ShaderMaterial
  private shadowMat!: THREE.ShaderMaterial
  private grassMesh!: THREE.InstancedMesh
  private scrubMesh!: THREE.InstancedMesh
  private sampleHeight!: (x: number, z: number) => number
  private moisture!: (x: number, z: number) => number
  /** Low-frequency clump field for the scrub ring, 0..1. */
  private scrubClump!: (x: number, z: number) => number
  /** Where the grass ring was last scattered. */
  private grassX = Infinity
  private grassZ = Infinity
  /** Where the scrub ring was last scattered. */
  private scrubX = Infinity
  private scrubZ = Infinity
  // Scratch objects — reused every frame / rebuild, never reallocated.
  private readonly camPos = new THREE.Vector3()
  private readonly sM = new THREE.Matrix4()
  private readonly sP = new THREE.Vector3()
  private readonly sQ = new THREE.Quaternion()
  private readonly sE = new THREE.Euler()
  private readonly sS = new THREE.Vector3()
  private readonly sC = new THREE.Color()

  /** Grows the archetypes, scatters the world, builds every mesh. */
  init(ctx: GameContext): void {
    const sample = ctx.getTerrainHeight
    this.sampleHeight = sample

    // --- Field masks -----------------------------------------------------
    const noise2 = createNoise2D(mulberry32(0xf1e1d))
    const fbm = (x: number, z: number, oct: number): number => {
      let s = 0
      let amp = 0.5
      let norm = 0
      let f = 1
      for (let i = 0; i < oct; i++) {
        s += amp * noise2(x * f, z * f)
        norm += amp
        amp *= 0.5
        f *= 2.03
      }
      return s / norm
    }
    /** Terrain gradient magnitude (rise over run) at 9 m spacing. */
    const slopeAt = (x: number, z: number): number => {
      const d = 9
      const dx = sample(x + d, z) - sample(x - d, z)
      const dz = sample(x, z + d) - sample(x, z - d)
      return Math.hypot(dx, dz) / (2 * d)
    }
    /** 0 = bone dry, 1 = riverbank. Drives density, species and grass tint. */
    const moisture = (x: number, z: number): number => {
      const h = sample(x, z)
      const riverM = Math.exp(-(x * x) / (850 * 850)) * sstep(-4400, -3800, z)
      const coastM = sstep(600, 2300, z)
      const plateau = sstep(2100, 2800, x)
      const lowland = 1 - sstep(150, 430, h)
      const base = 0.3 + 0.25 * fbm(x * 0.0006, z * 0.0006, 3)
      return sat(base * lowland + 0.55 * riverM + 0.22 * coastM * lowland - 0.42 * plateau)
    }
    this.moisture = moisture
    // Clump field for the scrub ring: ~140 m blobs so shrubs gather into
    // patches with bare ground between them instead of an even sprinkle.
    this.scrubClump = (x: number, z: number): number => 0.5 + 0.5 * fbm(x * 0.0045 + 13.7, z * 0.0045, 2)
    /**
     * Trees per m². Zero underwater, on cliffs and on the high tops.
     * Two THRESHOLDED noise scales structure the forest instead of an even
     * scatter: a broad belt mask (~600 m wavelength) decides where woodland
     * exists at all, and a fine copse mask (~140 m) breaks each belt into
     * tight clumps with dense cores and open gaps between them. Moisture
     * feeds both thresholds, so the copses chain into riparian ribbons along
     * the river and drainage lines while dry country falls apart into
     * isolated groves with lone-tree stragglers.
     */
    const treeDensity = (x: number, z: number, h: number, slope: number, moist: number): number => {
      if (h < 3 || h > 560 || slope > 0.45) return 0
      const plateau = sstep(2100, 2800, x)
      const n = 0.5 + 0.5 * fbm(x * 0.0016 + 40.5, z * 0.0016, 3)
      const belt = 0.03 + sstep(0.4, 0.66, n * 0.78 + moist * 0.45)
      const c = 0.5 + 0.5 * fbm(x * 0.0072 + 9.7, z * 0.0072, 2)
      const copse = 0.06 + 1.9 * sstep(0.46, 0.8, c * 0.72 + moist * 0.4)
      return (0.00003 + 0.00092 * Math.pow(moist, 1.5)) * (1 - plateau * 0.82) * belt * copse
    }

    // --- Textures + materials -------------------------------------------
    const foliageTex = toTexture(paintFoliageAtlas(), ctx.renderer)
    const grassTex = toTexture(paintGrassTexture(), ctx.renderer)
    const specs = makeSpecs()
    const impostorTex = toTexture(paintImpostorAtlas(specs), ctx.renderer)

    const barkMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0 })
    patchWind(barkMat, this.uTime, 'aviary-bark-v1')
    const foliageMat = new THREE.MeshStandardMaterial({
      map: foliageTex,
      vertexColors: true,
      alphaTest: 0.35,
      side: THREE.DoubleSide,
      roughness: 0.8,
      metalness: 0,
    })
    patchWind(foliageMat, this.uTime, 'aviary-foliage-v3', true, this.uSunDir)
    const depthMat = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
      map: foliageTex,
      alphaTest: 0.35,
    })
    patchWind(depthMat, this.uTime, 'aviary-veg-depth-v1')

    const archetypes = specs.map((s, i) => growTree(s, 0x7ee5 + i * 131071))

    const group = new THREE.Group()
    group.name = 'vegetation'
    group.matrixAutoUpdate = false

    // --- Tree scatter, chunk by chunk -----------------------------------
    /** One accepted tree: transform + tint, bucketed per archetype. */
    const impPos: number[] = []
    const impDim: number[] = []
    const impTint: number[] = []
    const impCol: number[] = []
    const impNorm: number[] = []

    for (let cj = 0; cj < CHUNKS; cj++) {
      for (let ci = 0; ci < CHUNKS; ci++) {
        const rng = mulberry32(0x5eed0 + cj * 8191 + ci * 127)
        const x0 = -HALF + ci * CHUNK_SIZE
        const z0 = -HALF + cj * CHUNK_SIZE
        const perArch: {
          x: number
          y: number
          z: number
          s: number
          yaw: number
          tilt: number
          tint: number
          hue: number
          variant: number
          nx: number
          ny: number
          nz: number
        }[][] = [[], [], [], []]
        let accepted = 0
        for (let n = 0; n < TREE_CANDIDATES && accepted < TREE_CHUNK_CAP; n++) {
          const x = x0 + rng() * CHUNK_SIZE
          const z = z0 + rng() * CHUNK_SIZE
          const h = sample(x, z)
          if (h < 3) continue
          // Terrain gradient → slope magnitude AND the ground normal the
          // blob-shadow decal needs to hug the local tangent plane.
          const gd = 9
          const ghx = (sample(x + gd, z) - sample(x - gd, z)) / (2 * gd)
          const ghz = (sample(x, z + gd) - sample(x, z - gd)) / (2 * gd)
          const slope = Math.hypot(ghx, ghz)
          const gInv = 1 / Math.sqrt(ghx * ghx + 1 + ghz * ghz)
          const moist = moisture(x, z)
          const dens = treeDensity(x, z, h, slope, moist)
          if (rng() >= dens * ((CHUNK_SIZE * CHUNK_SIZE) / TREE_CANDIDATES)) continue
          const plateau = sstep(2100, 2800, x)
          const riverM = Math.exp(-(x * x) / (850 * 850)) * sstep(-4400, -3800, z)
          const r = rng()
          let arch: number
          if (plateau > 0.5) arch = r < 0.85 ? 3 : 1
          else if (riverM > 0.5 && h < 70) arch = r < 0.8 ? 0 : 2
          else if (h > 220 || slope > 0.28) arch = r < 0.55 ? 1 : 2
          else if (moist > 0.5) arch = r < 0.6 ? 2 : 0
          else arch = r < 0.45 ? 1 : r < 0.85 ? 2 : 3
          const big = arch === 0 && riverM > 0.5
          // Scale jitter 0.6–1.6x (river red gums bias large) breaks the
          // single-scale treeline.
          const s = big ? 1.05 + rng() * 0.55 : 0.6 + Math.pow(rng(), 1.35)
          // Stand color: LOW-FREQUENCY noise fields (~900 m and ~1200 m
          // wavelengths) drive hue and lightness, so neighbouring trees share
          // a palette and whole hillsides drift warm/cool together. Per-tree
          // jitter is a whisper on top, and both channels are clamped to a
          // narrow band — no more lime-green tree beside a near-black one.
          const standHue = fbm(x * 0.0011 + 77.7, z * 0.0011, 2)
          const standL = fbm(x * 0.00085 - 31.3, z * 0.00085, 2)
          const hue = Math.max(-0.12, Math.min(0.12, standHue * 0.16 + (rng() - 0.5) * 0.05))
          const tint = Math.max(0.84, Math.min(1.16, 1 + standL * 0.22 + (rng() - 0.5) * 0.1))
          perArch[arch].push({
            x,
            y: h - 0.25,
            z,
            s,
            yaw: rng() * Math.PI * 2,
            tilt: (rng() - 0.5) * 0.1,
            tint,
            hue,
            variant: rng() < 0.5 ? 0 : 1,
            nx: -ghx * gInv,
            ny: gInv,
            nz: -ghz * gInv,
          })
          accepted++
        }
        const rec: VegChunk = {
          cx: x0 + CHUNK_SIZE / 2,
          cz: z0 + CHUNK_SIZE / 2,
          meshes: [],
          visible: false,
          shadows: false,
        }
        for (let a = 0; a < archetypes.length; a++) {
          const list = perArch[a]
          if (list.length === 0) continue
          const arch = archetypes[a]
          const mesh = new THREE.InstancedMesh(arch.geometry, [barkMat, foliageMat], list.length)
          for (let i = 0; i < list.length; i++) {
            const t = list[i]
            this.sE.set(t.tilt, t.yaw, 0, 'YXZ')
            this.sQ.setFromEuler(this.sE)
            this.sP.set(t.x, t.y, t.z)
            this.sS.setScalar(t.s)
            this.sM.compose(this.sP, this.sQ, this.sS)
            mesh.setMatrixAt(i, this.sM)
            // Clamped lightness 0.84–1.16 with a stand-clustered warm/cool
            // hue swing: positive hue pushes olive, negative blue-green.
            this.sC.setRGB(t.tint * (1 + t.hue * 1.2), t.tint, t.tint * (1 - t.hue * 1.5))
            mesh.setColorAt(i, this.sC)
            // Every tree also feeds the far impostor + blob-shadow sheets.
            impPos.push(t.x, t.y, t.z)
            impDim.push(arch.radius * 2 * t.s, arch.height * 1.04 * t.s)
            impTint.push(this.sC.r, this.sC.g, this.sC.b)
            impCol.push(a * 2 + t.variant)
            impNorm.push(t.nx, t.ny, t.nz)
          }
          mesh.customDepthMaterial = depthMat
          mesh.castShadow = false
          mesh.receiveShadow = true
          mesh.visible = false
          mesh.matrixAutoUpdate = false
          mesh.computeBoundingSphere()
          group.add(mesh)
          rec.meshes.push(mesh)
        }
        this.chunks.push(rec)
      }
    }

    // --- Far impostor sheet (one draw call for the whole forest) --------
    const iPosAttr = new THREE.InstancedBufferAttribute(new Float32Array(impPos), 3)
    const iDimAttr = new THREE.InstancedBufferAttribute(new Float32Array(impDim), 2)
    const impGeo = makeImpostorGeometry()
    impGeo.setAttribute('iPos', iPosAttr)
    impGeo.setAttribute('iDim', iDimAttr)
    impGeo.setAttribute('iTint', new THREE.InstancedBufferAttribute(new Float32Array(impTint), 3))
    impGeo.setAttribute('iCol', new THREE.InstancedBufferAttribute(new Float32Array(impCol), 1))
    impGeo.instanceCount = impCol.length
    this.impostorMat = makeImpostorMaterial(impostorTex, this.uAltBoost)
    const impostors = new THREE.Mesh(impGeo, this.impostorMat)
    impostors.frustumCulled = false
    impostors.matrixAutoUpdate = false
    group.add(impostors)

    // --- Blob-shadow decal sheet (one draw call, shares the buffers) ----
    const shadowGeo = new THREE.InstancedBufferGeometry()
    shadowGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, 0, -0.5, 0.5, 0, -0.5, 0.5, 0, 0.5, -0.5, 0, 0.5], 3),
    )
    shadowGeo.setIndex([0, 2, 1, 0, 3, 2])
    shadowGeo.setAttribute('iPos', iPosAttr)
    shadowGeo.setAttribute('iDim', iDimAttr)
    shadowGeo.setAttribute('iNorm', new THREE.InstancedBufferAttribute(new Float32Array(impNorm), 3))
    shadowGeo.instanceCount = impCol.length
    this.shadowMat = makeBlobShadowMaterial(this.uAltBoost)
    const blobShadows = new THREE.Mesh(shadowGeo, this.shadowMat)
    blobShadows.frustumCulled = false
    blobShadows.matrixAutoUpdate = false
    group.add(blobShadows)

    // --- Granite boulders ------------------------------------------------
    const graniteMat = makeGraniteMaterial()
    const rockGeos = [makeBoulderGeometry(0xa0c4, 0.68), makeBoulderGeometry(0xb1d5, 0.85)]
    const rockLists: { x: number; y: number; z: number; s: number; yaw: number }[][] = [[], []]
    const rockRng = mulberry32(0x40c5)
    for (let cj = 0; cj < CHUNKS; cj++) {
      for (let ci = 0; ci < CHUNKS; ci++) {
        for (let n = 0; n < ROCK_CANDIDATES; n++) {
          const x = -HALF + (ci + rockRng()) * CHUNK_SIZE
          const z = -HALF + (cj + rockRng()) * CHUNK_SIZE
          const h = sample(x, z)
          if (h < 1.5) continue
          const slope = slopeAt(x, z)
          const upland = sstep(180, 420, h)
          const scarp = sstep(2200, 2700, x) * (1 - sstep(3100, 3600, x))
          const p = 0.02 + 0.3 * sstep(0.2, 0.5, slope) + 0.2 * upland + 0.25 * scarp
          if (slope > 0.85 || rockRng() >= p * 0.5) continue
          const s = 0.9 + Math.pow(rockRng(), 2.2) * 5.5
          rockLists[rockRng() < 0.5 ? 0 : 1].push({ x, y: h - s * 0.28, z, s, yaw: rockRng() * Math.PI * 2 })
        }
      }
    }
    for (let v = 0; v < 2; v++) {
      const list = rockLists[v]
      if (list.length === 0) continue
      const mesh = new THREE.InstancedMesh(rockGeos[v], graniteMat, list.length)
      for (let i = 0; i < list.length; i++) {
        const t = list[i]
        this.sE.set(0, t.yaw, (hash2(i, v) - 0.5) * 0.24, 'YXZ')
        this.sQ.setFromEuler(this.sE)
        this.sP.set(t.x, t.y, t.z)
        this.sS.set(t.s * (0.8 + hash2(i, v + 7) * 0.5), t.s, t.s)
        this.sM.compose(this.sP, this.sQ, this.sS)
        mesh.setMatrixAt(i, this.sM)
      }
      mesh.castShadow = true
      mesh.receiveShadow = true
      mesh.matrixAutoUpdate = false
      mesh.computeBoundingSphere()
      group.add(mesh)
    }

    // --- Grass ring ------------------------------------------------------
    const grassMat = makeGroundCoverMaterial(
      grassTex,
      this.uTime,
      GRASS_RADIUS,
      GRASS_ALT_FADE,
      GRASS_MAX_ALT,
      this.uSunDir,
      this.uAlt,
      'aviary-grass-v4',
    )
    this.grassMesh = new THREE.InstancedMesh(makeCrossCardGeometry(1.2, 1.2, 3), grassMat, GRASS_CAPACITY)
    this.grassMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.grassMesh.frustumCulled = false
    this.grassMesh.castShadow = false
    this.grassMesh.receiveShadow = true
    this.grassMesh.matrixAutoUpdate = false
    group.add(this.grassMesh)

    // --- Scrub ring (saltbush/spinifex shrubs between the trees) ---------
    const scrubTex = toTexture(paintScrubTexture(), ctx.renderer)
    const scrubMat = makeGroundCoverMaterial(
      scrubTex,
      this.uTime,
      SCRUB_RADIUS,
      SCRUB_ALT_FADE,
      SCRUB_MAX_ALT,
      this.uSunDir,
      this.uAlt,
      'aviary-scrub-v2',
    )
    this.scrubMesh = new THREE.InstancedMesh(makeCrossCardGeometry(1.5, 1.05, 3), scrubMat, SCRUB_CAPACITY)
    this.scrubMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.scrubMesh.frustumCulled = false
    this.scrubMesh.castShadow = false
    this.scrubMesh.receiveShadow = true
    this.scrubMesh.matrixAutoUpdate = false
    group.add(this.scrubMesh)

    ctx.scene.add(group)

    // Settle everything around the spawn point before the first frame.
    this.camPos.set(WORLD.spawn.x, WORLD.spawn.y, WORLD.spawn.z)
    this.refreshChunks()
    this.scatterGrass(WORLD.spawn.x, WORLD.spawn.z)
    this.scatterScrub(WORLD.spawn.x, WORLD.spawn.z)
  }

  /** Advances wind, sun tint, chunk visibility and the ground-cover rings. */
  update(_dt: number, ctx: GameContext): void {
    this.uTime.value = ctx.time
    // Live sun for the foliage/ground-cover translucency term — without this
    // copy the wrap term lights from the frozen boot-time sun all day.
    this.uSunDir.value.copy(ctx.sunDirection)
    ;(this.impostorMat.uniforms.uSun.value as THREE.Vector3).copy(ctx.sunDirection)
    ;(this.shadowMat.uniforms.uSun.value as THREE.Vector3).copy(ctx.sunDirection)
    ctx.camera.getWorldPosition(this.camPos)
    this.refreshChunks()

    // refreshChunks just measured camera altitude over terrain into uAlt.
    const alt = this.uAlt.value
    const grassOn = alt < GRASS_MAX_ALT
    if (this.grassMesh.visible !== grassOn) this.grassMesh.visible = grassOn
    if (grassOn) {
      const dx = this.camPos.x - this.grassX
      const dz = this.camPos.z - this.grassZ
      if (dx * dx + dz * dz > GRASS_RESEED * GRASS_RESEED) this.scatterGrass(this.camPos.x, this.camPos.z)
    }
    const scrubOn = alt < SCRUB_MAX_ALT
    if (this.scrubMesh.visible !== scrubOn) this.scrubMesh.visible = scrubOn
    if (scrubOn) {
      const dx = this.camPos.x - this.scrubX
      const dz = this.camPos.z - this.scrubZ
      if (dx * dx + dz * dz > GRASS_RESEED * GRASS_RESEED) this.scatterScrub(this.camPos.x, this.camPos.z)
    }
  }

  /**
   * Walks all 256 chunk records: full geometry shows inside {@link TREE_NEAR}
   * (the impostor shader fades in at exactly that ring), shadow casting is
   * granted inside {@link TREE_SHADOW}. Both rings SHRINK with camera
   * altitude ({@link ALT_LOD_START} / {@link ALT_LOD_GAIN}): from the air,
   * alpha-tested leaf cards mip into salt-and-pepper speckle, so climbing
   * hands chunks over to the solid-silhouette impostor crowns instead. The
   * measured altitude also feeds the shared `uAlt` / `uAltBoost` uniforms,
   * so the impostor + blob-shadow shaders apply the SAME boost to their
   * distance tests (hand-over stays gap-free) and the ground-cover rings
   * scale-fade out over their altitude bands instead of popping off.
   * Zero allocations.
   */
  private refreshChunks(): void {
    const px = this.camPos.x
    const pz = this.camPos.z
    const alt = this.camPos.y - this.sampleHeight(px, pz)
    this.uAlt.value = alt
    const boost = Math.max(0, alt - ALT_LOD_START) * ALT_LOD_GAIN
    this.uAltBoost.value = boost
    const nearR = Math.max(0, TREE_NEAR - boost)
    const shadowR = Math.max(0, TREE_SHADOW - boost)
    const nearSq = nearR * nearR
    const shadowSq = shadowR * shadowR
    for (let i = 0; i < this.chunks.length; i++) {
      const rec = this.chunks[i]
      const dx = rec.cx - px
      const dz = rec.cz - pz
      const d2 = dx * dx + dz * dz
      const vis = d2 < nearSq
      const sh = d2 < shadowSq
      if (vis !== rec.visible || sh !== rec.shadows) {
        rec.visible = vis
        rec.shadows = sh
        for (let m = 0; m < rec.meshes.length; m++) {
          rec.meshes[m].visible = vis
          rec.meshes[m].castShadow = sh
        }
      }
    }
  }

  /**
   * Re-scatters the tussock ring around (cx, cz). Placement derives from a
   * world-space cell hash, so a spot on the ground always grows the same
   * tussock no matter which direction the bird approaches from. Runs only
   * when the camera has strayed {@link GRASS_RESEED} meters; writes straight
   * into the instance buffers with zero allocation.
   */
  private scatterGrass(cx: number, cz: number): void {
    this.grassX = cx
    this.grassZ = cz
    const i0 = Math.floor((cx - GRASS_RADIUS) / GRASS_CELL)
    const i1 = Math.ceil((cx + GRASS_RADIUS) / GRASS_CELL)
    const j0 = Math.floor((cz - GRASS_RADIUS) / GRASS_CELL)
    const j1 = Math.ceil((cz + GRASS_RADIUS) / GRASS_CELL)
    const r2 = GRASS_RADIUS * GRASS_RADIUS
    let count = 0
    for (let j = j0; j <= j1 && count < GRASS_CAPACITY; j++) {
      for (let i = i0; i <= i1 && count < GRASS_CAPACITY; i++) {
        const h0 = hash2(i, j)
        const x = (i + h0) * GRASS_CELL
        const z = (j + hash2(i, j + 611)) * GRASS_CELL
        const dx = x - cx
        const dz = z - cz
        if (dx * dx + dz * dz > r2) continue
        const y = this.sampleHeight(x, z)
        if (y < 2.2 || y > 540) continue
        const moist = this.moisture(x, z)
        // Generous base density: even bone-dry country keeps a scatter of
        // tussocks, so low flight always has near-field parallax to read
        // speed against (bare smeared ground kills the speed cue).
        const p = 0.24 + 0.62 * moist + 0.18 * hash2(i + 917, j)
        if (hash2(i + 331, j + 77) >= p) continue
        // Cheap slope probe: one extra height tap.
        if (Math.abs(this.sampleHeight(x + 7, z + 7) - y) > 4.2) continue
        const s = 0.55 + hash2(i + 13, j + 29) * (0.7 + moist * 0.9)
        this.sE.set(0, h0 * Math.PI * 2, 0)
        this.sQ.setFromEuler(this.sE)
        this.sP.set(x, y - 0.05, z)
        this.sS.set(s, s * (0.8 + hash2(i, j + 5) * 0.5), s)
        this.sM.compose(this.sP, this.sQ, this.sS)
        this.grassMesh.setMatrixAt(count, this.sM)
        // Straw-gold when dry, sap-green on the riverbanks.
        const g = sat(moist * 1.4 - 0.15)
        const b = 0.8 + hash2(i + 3, j + 9) * 0.4
        this.sC.setRGB(lerp(1.0, 0.55, g) * b, lerp(0.82, 0.72, g) * b, lerp(0.5, 0.34, g) * b)
        this.grassMesh.setColorAt(count, this.sC)
        count++
      }
    }
    this.grassMesh.count = count
    this.grassMesh.instanceMatrix.needsUpdate = true
    if (this.grassMesh.instanceColor) this.grassMesh.instanceColor.needsUpdate = true
  }

  /**
   * Re-scatters the scrub ring (saltbush/spinifex shrubs) around (cx, cz).
   * Same stable world-space cell hash as the grass, but on a wider cell with
   * clump-noise + moisture density, so shrubs gather into patches with open
   * ground between them — the low-altitude parallax speed cue the bare
   * terrain was missing. Zero allocation.
   */
  private scatterScrub(cx: number, cz: number): void {
    this.scrubX = cx
    this.scrubZ = cz
    const i0 = Math.floor((cx - SCRUB_RADIUS) / SCRUB_CELL)
    const i1 = Math.ceil((cx + SCRUB_RADIUS) / SCRUB_CELL)
    const j0 = Math.floor((cz - SCRUB_RADIUS) / SCRUB_CELL)
    const j1 = Math.ceil((cz + SCRUB_RADIUS) / SCRUB_CELL)
    const r2 = SCRUB_RADIUS * SCRUB_RADIUS
    let count = 0
    for (let j = j0; j <= j1 && count < SCRUB_CAPACITY; j++) {
      for (let i = i0; i <= i1 && count < SCRUB_CAPACITY; i++) {
        // Distinct hash streams from the grass ring so the two layers
        // never stack on the same spots.
        const h0 = hash2(i + 51, j + 23)
        const x = (i + h0) * SCRUB_CELL
        const z = (j + hash2(i + 407, j + 611)) * SCRUB_CELL
        const dx = x - cx
        const dz = z - cz
        if (dx * dx + dz * dz > r2) continue
        const y = this.sampleHeight(x, z)
        if (y < 2.2 || y > 620) continue
        const moist = this.moisture(x, z)
        const clump = this.scrubClump(x, z)
        // Clump-noise gate: dense hearts, sparse fringes — but keep a real
        // baseline everywhere so the ground between trees never reads bare.
        const p = 0.12 + 0.78 * sstep(0.4, 0.78, clump * 0.72 + moist * 0.33) + 0.1 * moist
        if (hash2(i + 733, j + 177) >= p) continue
        // Cheap slope probe: one extra height tap.
        if (Math.abs(this.sampleHeight(x + 7, z + 7) - y) > 4.2) continue
        const s = 0.7 + hash2(i + 63, j + 91) * (1.0 + moist * 0.5)
        this.sE.set(0, h0 * Math.PI * 2, 0)
        this.sQ.setFromEuler(this.sE)
        this.sP.set(x, y - 0.05, z)
        this.sS.set(s, s * (0.75 + hash2(i + 5, j + 47) * 0.45), s)
        this.sM.compose(this.sP, this.sQ, this.sS)
        this.scrubMesh.setMatrixAt(count, this.sM)
        // Dusty sage grey-green when dry, richer olive near water.
        const b = 0.82 + hash2(i + 19, j + 3) * 0.36
        this.sC.setRGB(
          lerp(0.68, 0.46, moist) * b,
          lerp(0.66, 0.56, moist) * b,
          lerp(0.5, 0.34, moist) * b,
        )
        this.scrubMesh.setColorAt(count, this.sC)
        count++
      }
    }
    this.scrubMesh.count = count
    this.scrubMesh.instanceMatrix.needsUpdate = true
    if (this.scrubMesh.instanceColor) this.scrubMesh.instanceColor.needsUpdate = true
  }
}
