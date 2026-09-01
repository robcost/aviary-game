/**
 * Terrain — the Australian landscape of Aviary.
 *
 * A 12 km × 12 km heightfield that honors the shared world layout in
 * {@link WORLD}: ocean to the south past z ≈ 3000, a dome-profiled mountain
 * range in the north-west quadrant peaking near 900 m, a meandering river
 * valley along x ≈ 0, and an arid, escarpment-edged plateau to the east.
 *
 * Architecture:
 * - The macro height function (layered simplex FBM, domain warping, rounded
 *   ridged noise, river carving, mesa terracing) is evaluated ONCE at init
 *   into a 1025 × 1025 base grid, then upsampled with the same triangle-split
 *   interpolation into the authoritative 2049 × 2049 fine grid (5.86 m cells)
 *   with slope- and shore-modulated detail octaves added. Every detail octave
 *   spans ≥ 3.8 grid cells, so the baked surface stays band-limited: no
 *   aliased triangle-soup facets near the camera.
 * - Both the rendered mesh and {@link GameContext.getTerrainHeight} sample
 *   the fine grid with the SAME triangle-split interpolation, so at the
 *   finest LOD the CPU sampler matches the rendered surface exactly.
 * - Geometry is a 16 × 16 grid of 750 m chunks. Each chunk lazily builds one
 *   of five LOD meshes (128/64/32/16/8 segments) with a dropped skirt ring
 *   that hides cracks at LOD boundaries. The 128-segment level is a 1.2 km
 *   near ring (full 5.86 m grid) and vertex spacing stays ≤ 11.7 m to
 *   2.6 km, so foreground and mid-distance cliff silhouettes stay smooth;
 *   coarser LODs box-filter the heightfield to their own vertex
 *   spacing, so distant ridges keep their dome silhouettes instead of
 *   aliasing into needle spikes. LOD follows the camera every frame with
 *   hysteresis, at zero steady-state allocation.
 * - Shading never depends on mesh resolution: a 2049² world-normal texture is
 *   baked from the fine grid and sampled per pixel (mipmapped + anisotropic),
 *   so no LOD ever shows flat-shaded facets.
 * - One shared MeshStandardMaterial, hooked via onBeforeCompile, splats
 *   ochre earth / spinifex / scrub / sandstone strata / beach sand by height,
 *   slope and noise. All cell-forming patterns are triplanar-projected (no
 *   stretching on cliffs) and cross-faded by km-scale macro noise (no
 *   world-scale tiling). A hemisphere fill term keeps shadowed faces from
 *   going black. Shadows and scene environment keep working because the
 *   standard lighting chunks are left intact.
 * - On top of the macro splat sit three baked, tileable, world-space
 *   triplanar DETAIL cascades (albedo + normal, ~1.7 m / ~13 m / ~65 m
 *   tiles, scale-drifted by the macro fields so they never read as a
 *   repeating stamp) with separate soil/grass/rock channels; the rock
 *   channel is selected by the slope-thresholded cliff mask so escarpment
 *   faces read as cracked, bedded stone. NO tier ever distance-fades: every
 *   channel centers on 0.5, so mip minification converges to zero modulation
 *   on its own with no shimmer, and the ground keeps its finest resolvable
 *   texture octave at EVERY altitude and range instead of dissolving into
 *   flat color splats past a hard-coded fade distance. (This also keeps all
 *   mipmapped detail samples in uniform control flow — no undefined
 *   derivatives at fade-band edges.)
 * - The shader reads terrain height from a baked half-float height texture,
 *   not from interpolated vertex position: coarse LODs render box-filtered
 *   geometry whose height differs from the fine grid by meters, and reading
 *   vWorldPos.y used to flip the height-gated splat masks along straight
 *   chunk-border lines (rectangular "tile" patches). The texture height is
 *   identical at every LOD, so no seam can exist. All splat mask thresholds
 *   are additionally dithered by mid-scale noise over a multi-meter band.
 * - Above ~35° of slope the km-scale macro fields swap from top-down xz
 *   projection to triplanar, and the top-down procedural detail gradients
 *   fade out entirely — steep canyon walls carry plane-projected detail
 *   only, never vertically smeared top-down patterns.
 * - Slope-driven masks are distance-compensated (mip-filtered normals read
 *   flatter at range), and a ~120 m triplanar macro-variation band plus
 *   strata modulation on the high-country cap persist at ALL distances, so
 *   far mountains keep material variation instead of fading to flat tan.
 * - Familiar-scale anchors: two static InstancedMeshes of termite mounds
 *   (1.8–4.5 m, plains and foothill flats, sharing a magnetic N–S long axis)
 *   and dead eucalypt snags (7–13 m, along the river corridor) give the
 *   midground a known real-world dimension so altitude reads from parallax.
 * - The terrain's fog chunk is replaced with height-based exponential aerial
 *   perspective: density decays with altitude (peaks rise out of valley
 *   haze), and the haze color runs warm at the horizon line and cools toward
 *   sky blue on downward rays. Three procedural ridgeline silhouette rings
 *   beyond the world edge (9.6 / 13.6 / 19.2 km, tinted live from the scene
 *   fog color, progressively lighter with distance) layer the horizon so
 *   distance no longer compresses into one flat haze band.
 */
import * as THREE from 'three'
import { createNoise2D } from 'simplex-noise'
import { WORLD, type GameContext, type GameModule } from '../core/GameState'

// ---------------------------------------------------------------------------
// Grid + chunk constants
// ---------------------------------------------------------------------------

/** Base (macro bake) heightfield resolution in cells per side. */
const BASE_GRID = 1024
/** Base heightfield samples per side. */
const BASE_N = BASE_GRID + 1
/** World size of one base cell, meters (~11.72). */
const BASE_CELL = WORLD.size / BASE_GRID
/** Authoritative fine heightfield resolution in cells per side. */
const GRID = 2048
/** Fine heightfield samples per side (grid corners). */
const GRID_N = GRID + 1
/** Half the world edge, meters. */
const HALF = WORLD.size / 2
/** World size of one fine grid cell, meters (~5.86). */
const CELL = WORLD.size / GRID
/** Chunks per side. */
const CHUNKS = 16
/** World size of one chunk, meters (750). */
const CHUNK_SIZE = WORLD.size / CHUNKS
/** Fine grid cells per chunk side at the finest LOD (128). */
const CHUNK_SEGS = GRID / CHUNKS
/** Fine-grid-cell stride per LOD level. */
const LOD_STEP = [1, 2, 4, 8, 16] as const
/** Number of LOD levels. */
const LOD_COUNT = LOD_STEP.length
/**
 * Outer distance (meters) at which each LOD level hands over to the next.
 * The 1200 m LOD-0 ring guarantees the chunk under a low-flying bird (chunk
 * center is at most ~531 m away near ground level) always renders the exact
 * fine-grid surface the collision sampler returns, and keeps foreground
 * cliff silhouettes on the full 5.86 m grid well past the foreground. The
 * LOD-1/2 rings run to 2.6 / 5.2 km so mid-distance escarpment faces — the
 * classic art-review close-up — never show low-poly silhouette facets:
 * vertex spacing stays ≤ 11.7 m to 2.6 km and ≤ 23.4 m to 5.2 km.
 */
const LOD_DIST = [1200, 2600, 5200, 9200] as const

// ---------------------------------------------------------------------------
// Deterministic noise toolkit (CPU side)
// ---------------------------------------------------------------------------

/**
 * Mulberry32 PRNG. Gives the simplex permutation a fixed seed so every boot
 * generates the identical continent (flight physics, vegetation placement and
 * photo shots all depend on reproducibility).
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
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

// ---------------------------------------------------------------------------
// Heightfield generation
// ---------------------------------------------------------------------------

/**
 * Bakes the full landscape into the authoritative row-major
 * {@link GRID_N} × {@link GRID_N} fine Float32Array (index = iz * GRID_N + ix).
 *
 * Pass 1 (base, 1025²): coastline ramp, domain-warped FBM hills, the NW
 * mountain range (rounded ridged noise pushed through a beehive-dome profile
 * with a hard amplitude clamp, so no needle spikes can survive), terraced
 * eastern plateau, carved river valley, two fine octaves.
 *
 * Pass 2 (fine, 2049²): triangle-split upsample of the base grid plus three
 * detail octaves (77 / 37 / 22 m wavelengths — every octave spans ≥ 3.8 fine
 * cells, so the 5.86 m grid samples them cleanly and the surface never turns
 * into aliased triangle soup). Detail amplitude grows on steep rocky ground
 * and fades on beaches and the seabed so sand stays smooth. The fine grid is
 * the single source of truth for the sampler, the meshes and the normal
 * texture; sub-22 m relief lives purely in the shader's detail normals.
 */
function generateHeightField(): Float32Array {
  const noise = createNoise2D(mulberry32(0xa551e5))

  /** Signed FBM in roughly -1..1. */
  const fbm = (x: number, z: number, octaves: number): number => {
    let sum = 0
    let amp = 0.5
    let norm = 0
    let f = 1
    for (let i = 0; i < octaves; i++) {
      sum += amp * noise(x * f, z * f)
      norm += amp
      amp *= 0.5
      f *= 2.02
    }
    return sum / norm
  }

  /**
   * Rounded ridged multifractal in roughly 0..1. Each octave runs
   * `1 - |noise|` through a smoothstep, which zeroes the derivative at both
   * the crest and the valley floor: crests come out as rounded domes instead
   * of the classic knife-edge (and, after masking, needle) artifact.
   */
  const ridged = (x: number, z: number, octaves: number): number => {
    let sum = 0
    let amp = 0.5
    let prev = 1
    let f = 1
    for (let i = 0; i < octaves; i++) {
      let n = 1 - Math.abs(noise(x * f, z * f))
      n = n * n * (3 - 2 * n)
      sum += n * amp * prev
      prev = n
      amp *= 0.5
      f *= 2.1
    }
    return sum
  }

  // Per-column coastline (depends on x only) and per-row river path
  // (depends on z only) are hoisted out of the main loop.
  const coastZ = new Float32Array(BASE_N)
  for (let ix = 0; ix < BASE_N; ix++) {
    const x = -HALF + ix * BASE_CELL
    coastZ[ix] = 3000 + 480 * fbm(x * 0.00033, 4.7, 3)
  }
  const riverX = new Float32Array(BASE_N)
  const riverBed = new Float32Array(BASE_N)
  for (let iz = 0; iz < BASE_N; iz++) {
    const z = -HALF + iz * BASE_CELL
    riverX[iz] = 320 * Math.sin(z * 0.00085 + 0.6) + 430 * fbm(z * 0.00045, 55.5, 3)
    const t = Math.min(1, Math.max(0, (3000 - z) / 7000))
    riverBed[iz] = 1.5 + 145 * t * t
  }

  // --- Pass 1: macro landscape on the base grid ---
  const base = new Float32Array(BASE_N * BASE_N)
  for (let iz = 0; iz < BASE_N; iz++) {
    const z = -HALF + iz * BASE_CELL
    const rvX = riverX[iz]
    const rvBed = riverBed[iz]
    const rowRiverOn = sstep(-4600, -4000, z) // the river rises in the ranges
    for (let ix = 0; ix < BASE_N; ix++) {
      const x = -HALF + ix * BASE_CELL

      // --- Continental base: ocean shelf south, rising country inland ---
      const d = coastZ[ix] - z // inland distance from the wiggling coastline
      const shore = sstep(-380, 520, d)
      const ocean = -8 - 54 * sstep(120, 2200, -d)
      let h = lerp(ocean, 34 + 74 * sstep(400, 5600, d), shore)

      // --- Domain warp shared by the hill and mountain octaves ---
      const wx = x + 720 * fbm(x * 0.00021 + 13.1, z * 0.00021, 3)
      const wz = z + 720 * fbm(x * 0.00021 - 7.7, z * 0.00021 + 71.3, 3)

      // --- Rolling hill country ---
      h += shore * (56 * fbm(wx * 0.00052, wz * 0.00052, 5) + 13 * fbm(wx * 0.0023, wz * 0.0023, 3))

      // --- NW mountain range: rounded ridges under a rotated gaussian mask,
      // reshaped into beehive domes (Bungle Bungle profile) ---
      const mu = (x - z) * 0.70710678 // NE-SW long axis through (-3500,-3500)
      const mv = (x + 3500 + (z + 3500)) * 0.70710678
      const mq = (mu * mu) / (2 * 2700 * 2700) + (mv * mv) / (2 * 1500 * 1500)
      if (mq < 9) {
        const mask = Math.exp(-mq)
        // Knob-modulate the crest lines so continuous ridges pinch into rows
        // of individual domes.
        let r = ridged(wx * 0.00092, wz * 0.00092, 5) * (0.86 + 0.14 * fbm(wx * 0.0019 + 91.2, wz * 0.0019, 2))
        if (r > 1) r = 1
        else if (r < 0) r = 0
        // Beehive-dome profile: height = maxH * (1 - u²)^1.5 with u = 1 - r.
        // Zero derivative at the top rounds every summit; the clamp above
        // hard-caps amplitude so no octave stack can spike past maxH.
        const u = 1 - r
        const dome = Math.pow(1 - u * u, 1.5)
        h += mask * shore * (110 + 640 * dome)
      }

      // --- Eastern plateau: flat-topped mesa country behind an escarpment ---
      if (x > 1500) {
        const edge = 340 * fbm(z * 0.00042 + 9.9, x * 0.00009, 3)
        const pm = sstep(2150, 2950, x + edge) * shore
        if (pm > 0.001) {
          const top = 235 + 48 * fbm(x * 0.0011 + 3.3, z * 0.0011, 4)
          h = lerp(h, top, pm)
          // Terrace the escarpment band into sandstone benches.
          const band = pm * (1 - pm) * 4
          h = lerp(h, Math.round(h / 26) * 26, 0.5 * band)
        }
      }

      // --- River valley: carve toward a graded bed along the meander ---
      const dr = Math.abs(x - rvX)
      if (dr < 950) {
        const vAmt = (1 - sstep(70, 900, dr)) * sstep(-300, 300, d) * rowRiverOn
        if (vAmt > 0.001) {
          const channel = 1 - sstep(0, 120, dr)
          const target = rvBed + 5 - 9 * channel * channel
          h = lerp(h, Math.min(h, target), vAmt)
        }
      }

      // --- Fine relief so the ground reads at flight altitude ---
      h += shore * (6 * fbm(x * 0.0058, z * 0.0058, 3) + 2.2 * fbm(x * 0.019 + 31.7, z * 0.019, 2))

      base[iz * BASE_N + ix] = h
    }
  }

  // --- Pass 2: upsample to the fine grid and add near-field detail ---
  const fine = new Float32Array(GRID_N * GRID_N)
  for (let iz = 0; iz < GRID_N; iz++) {
    const z = -HALF + iz * CELL
    const bz = iz * 0.5
    let bzi = bz | 0
    if (bzi >= BASE_GRID) bzi = BASE_GRID - 1
    const fz = bz - bzi
    for (let ix = 0; ix < GRID_N; ix++) {
      const x = -HALF + ix * CELL
      const bx = ix * 0.5
      let bxi = bx | 0
      if (bxi >= BASE_GRID) bxi = BASE_GRID - 1
      const fx = bx - bxi
      const b = bzi * BASE_N + bxi
      const hA = base[b]
      const hB = base[b + 1]
      const hC = base[b + BASE_N]
      const hD = base[b + BASE_N + 1]
      let h = fx > fz ? hA + (hB - hA) * fx + (hD - hB) * fz : hA + (hC - hA) * fz + (hD - hC) * fx

      // Detail relief: rougher on steep rocky ground, faded flat on the
      // beach and seabed so sand and shallows stay clean.
      const dhx = ((hB - hA + hD - hC) * 0.5) / BASE_CELL
      const dhz = ((hC - hA + hD - hB) * 0.5) / BASE_CELL
      const rough = 0.55 + 0.9 * sstep(0.22, 0.6, Math.sqrt(dhx * dhx + dhz * dhz))
      const dAmp = (0.2 + 0.8 * sstep(2.0, 7.0, h)) * rough
      let dh = 1.6 * noise(x * 0.013 + 5.2, z * 0.013)
      dh += 0.75 * noise(x * 0.027 + 71.3, z * 0.027)
      dh += 0.35 * noise(x * 0.045, z * 0.045 + 11.8)
      fine[iz * GRID_N + ix] = h + dAmp * dh
    }
  }
  return fine
}

// ---------------------------------------------------------------------------
// Exact CPU sampler
// ---------------------------------------------------------------------------

/**
 * Builds the {@link GameContext.getTerrainHeight} sampler. It interpolates
 * the fine grid with the SAME diagonal split (the a→d diagonal, i.e. the
 * fx = fz line) used by the chunk index buffers, so at LOD 0 the returned
 * height lies exactly on the rendered triangles. Costs ~15 arithmetic ops
 * per call and allocates nothing.
 */
function makeSampler(grid: Float32Array): (x: number, z: number) => number {
  const inv = 1 / CELL
  return (x: number, z: number): number => {
    let gx = (x + HALF) * inv
    let gz = (z + HALF) * inv
    if (gx < 0) gx = 0
    else if (gx > GRID) gx = GRID
    if (gz < 0) gz = 0
    else if (gz > GRID) gz = GRID
    let ix = gx | 0
    let iz = gz | 0
    if (ix >= GRID) ix = GRID - 1
    if (iz >= GRID) iz = GRID - 1
    const fx = gx - ix
    const fz = gz - iz
    const base = iz * GRID_N + ix
    const hA = grid[base]
    const hB = grid[base + 1]
    const hC = grid[base + GRID_N]
    const hD = grid[base + GRID_N + 1]
    return fx > fz
      ? hA + (hB - hA) * fx + (hD - hB) * fz // triangle A-D-B
      : hA + (hC - hA) * fz + (hD - hC) * fx // triangle A-C-D
  }
}

// ---------------------------------------------------------------------------
// Baked world-normal texture
// ---------------------------------------------------------------------------

/**
 * Derives a world-space normal map from the fine heightfield (central
 * differences, encoded n * 0.5 + 0.5 into RGBA8). The fragment shader samples
 * it per pixel with trilinear + anisotropic filtering, which makes shading
 * independent of mesh LOD: no vertex-normal facets near the camera and no
 * chunky lighting on distant low-LOD chunks.
 */
function bakeNormalTexture(grid: Float32Array): THREE.DataTexture {
  const data = new Uint8Array(GRID_N * GRID_N * 4)
  let w = 0
  for (let iz = 0; iz < GRID_N; iz++) {
    const zm = iz > 0 ? iz - 1 : 0
    const zp = iz < GRID ? iz + 1 : GRID
    const rowM = zm * GRID_N
    const rowP = zp * GRID_N
    const row = iz * GRID_N
    const dz = (zp - zm) * CELL
    for (let ix = 0; ix < GRID_N; ix++) {
      const xm = ix > 0 ? ix - 1 : 0
      const xp = ix < GRID ? ix + 1 : GRID
      const dhdx = (grid[row + xp] - grid[row + xm]) / ((xp - xm) * CELL)
      const dhdz = (grid[rowP + ix] - grid[rowM + ix]) / dz
      const nl = 1 / Math.sqrt(dhdx * dhdx + 1 + dhdz * dhdz)
      data[w++] = (255 * (-dhdx * nl * 0.5 + 0.5)) | 0
      data[w++] = (255 * (nl * 0.5 + 0.5)) | 0
      data[w++] = (255 * (-dhdz * nl * 0.5 + 0.5)) | 0
      data[w++] = 255
    }
  }
  const tex = new THREE.DataTexture(data, GRID_N, GRID_N, THREE.RGBAFormat, THREE.UnsignedByteType)
  tex.wrapS = THREE.ClampToEdgeWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.generateMipmaps = true
  tex.colorSpace = THREE.NoColorSpace
  tex.needsUpdate = true
  return tex
}

// ---------------------------------------------------------------------------
// Baked tileable detail textures (albedo modulation + bump normals)
// ---------------------------------------------------------------------------

/** Detail texture resolution per side (power of two: mipmapped + repeated). */
const DET_SIZE = 512
/** Soil bump-normal gradient gain (per-texel height delta → slope). */
const SOIL_BUMP = 7.0
/** Rock bump-normal gradient gain. Rock reads harder than loam. */
const ROCK_BUMP = 11.0

/** Integer lattice hash for tileable noise, result in [0, 1). */
function latHash(ix: number, iy: number, seed: number): number {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(seed, 1013904223)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  h ^= h >>> 16
  return (h >>> 0) / 4294967296
}

/**
 * Tileable value noise: the lattice wraps at period `p`, so a (u, v) ∈ [0, 1)
 * tile is perfectly seamless under RepeatWrapping.
 */
function tnoise(u: number, v: number, p: number, seed: number): number {
  const xf = u * p
  const zf = v * p
  const xi = Math.floor(xf)
  const zi = Math.floor(zf)
  const fx = xf - xi
  const fz = zf - zi
  const x0 = ((xi % p) + p) % p
  const z0 = ((zi % p) + p) % p
  const x1 = (x0 + 1) % p
  const z1 = (z0 + 1) % p
  const ux = fx * fx * (3 - 2 * fx)
  const uz = fz * fz * (3 - 2 * fz)
  const a = latHash(x0, z0, seed)
  const b = latHash(x1, z0, seed)
  const c = latHash(x0, z1, seed)
  const d = latHash(x1, z1, seed)
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz
}

/** Tileable FBM built on {@link tnoise}. Result roughly in 0..1. */
function tfbm(u: number, v: number, p0: number, octaves: number, seed: number): number {
  let sum = 0
  let amp = 0.5
  let norm = 0
  let p = p0
  for (let i = 0; i < octaves; i++) {
    sum += amp * tnoise(u, v, p, seed + i * 131)
    norm += amp
    amp *= 0.5
    p *= 2
  }
  return sum / norm
}

/** The two baked detail maps the terrain shader samples triplanar. */
interface DetailTextures {
  /** R = soil albedo mod, G = grass, B = rock — all centered on 0.5. */
  albedo: THREE.DataTexture
  /** RG = soil bump normal xy, BA = rock bump normal xy (n * 0.5 + 0.5). */
  normal: THREE.DataTexture
}

/**
 * Bakes the seamless per-material detail maps once at init. Soil is granular
 * loam with sparse embedded pebbles, grass is clumped spinifex tufts over
 * matted ground, rock is bedded stone broken by two crack networks. Each
 * channel is normalized around its own mean so the shader can treat 0.5 as
 * "no modulation"; bump normals come from wrapped central differences on the
 * same height fields.
 */
function bakeDetailTextures(): DetailTextures {
  const n = DET_SIZE
  const soil = new Float32Array(n * n)
  const grass = new Float32Array(n * n)
  const rock = new Float32Array(n * n)
  let w = 0
  for (let y = 0; y < n; y++) {
    const v = y / n
    for (let x = 0; x < n; x++, w++) {
      const u = x / n
      // Soil: granular loam + pebbles pushed up out of the grain.
      let hs = tfbm(u, v, 6, 5, 11) + 0.32 * tfbm(u, v, 28, 3, 23)
      hs += 0.55 * sstep(0.76, 0.9, tnoise(u, v, 34, 37))
      soil[w] = hs
      // Grass: tufts thresholded out of mid noise, high-frequency interior.
      const tuft = sstep(0.55, 0.85, tfbm(u, v, 26, 3, 53))
      grass[w] = 0.55 * tfbm(u, v, 11, 4, 47) + 0.6 * tuft * (0.4 + 0.6 * tnoise(u, v, 64, 59))
      // Rock: bedded mass minus two thin crack networks.
      const cr1 = Math.max(0, 1 - Math.abs(2 * tnoise(u, v, 7, 71) - 1) * 3.2)
      const cr2 = Math.max(0, 1 - Math.abs(2 * tnoise(u, v, 15, 83) - 1) * 3.6)
      rock[w] = 0.6 * tfbm(u, v, 5, 4, 89) + 0.34 * tfbm(u, v, 21, 3, 97) - 0.5 * cr1 * cr1 - 0.32 * cr2 * cr2
    }
  }

  /** Recenters a channel on 0.5 with the given contrast gain, clamped. */
  const recenter = (f: Float32Array, gain: number): void => {
    let m = 0
    for (let i = 0; i < f.length; i++) m += f[i]
    m /= f.length
    for (let i = 0; i < f.length; i++) {
      const t = 0.5 + (f[i] - m) * gain
      f[i] = t < 0.03 ? 0.03 : t > 0.97 ? 0.97 : t
    }
  }
  recenter(soil, 0.9)
  recenter(grass, 1.0)
  recenter(rock, 1.05)

  const albedo = new Uint8Array(n * n * 4)
  const normal = new Uint8Array(n * n * 4)
  for (let y = 0; y < n; y++) {
    const ym = ((y + n - 1) % n) * n
    const yp = ((y + 1) % n) * n
    const row = y * n
    for (let x = 0; x < n; x++) {
      const xm = (x + n - 1) % n
      const xp = (x + 1) % n
      const i = row + x
      const o = i * 4
      albedo[o] = (soil[i] * 255) | 0
      albedo[o + 1] = (grass[i] * 255) | 0
      albedo[o + 2] = (rock[i] * 255) | 0
      albedo[o + 3] = 255
      const sgx = (soil[row + xp] - soil[row + xm]) * SOIL_BUMP
      const sgy = (soil[yp + x] - soil[ym + x]) * SOIL_BUMP
      const snl = 1 / Math.sqrt(sgx * sgx + sgy * sgy + 1)
      const rgx = (rock[row + xp] - rock[row + xm]) * ROCK_BUMP
      const rgy = (rock[yp + x] - rock[ym + x]) * ROCK_BUMP
      const rnl = 1 / Math.sqrt(rgx * rgx + rgy * rgy + 1)
      normal[o] = (255 * (-sgx * snl * 0.5 + 0.5)) | 0
      normal[o + 1] = (255 * (-sgy * snl * 0.5 + 0.5)) | 0
      normal[o + 2] = (255 * (-rgx * rnl * 0.5 + 0.5)) | 0
      normal[o + 3] = (255 * (-rgy * rnl * 0.5 + 0.5)) | 0
    }
  }

  const mk = (data: Uint8Array): THREE.DataTexture => {
    const t = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType)
    t.wrapS = THREE.RepeatWrapping
    t.wrapT = THREE.RepeatWrapping
    t.magFilter = THREE.LinearFilter
    t.minFilter = THREE.LinearMipmapLinearFilter
    t.generateMipmaps = true
    t.colorSpace = THREE.NoColorSpace
    t.needsUpdate = true
    return t
  }
  return { albedo: mk(albedo), normal: mk(normal) }
}

// ---------------------------------------------------------------------------
// Chunk geometry
// ---------------------------------------------------------------------------

/**
 * Box-filtered heightfield sample. Radius 0 reads the fine grid directly;
 * radius r averages the (2r+1)² window (clamped at the borders). Coarse LODs
 * sample with r = step, a window twice the vertex spacing whose first
 * sinc null lands on the sampling Nyquist wavelength — point-sampling a
 * 5.86 m grid every 47–94 m is what turned distant dome country into needle
 * spikes and shimmering facet soup.
 */
function filteredHeight(grid: Float32Array, gx: number, gz: number, r: number): number {
  if (r === 0) return grid[gz * GRID_N + gx]
  const x0 = gx - r < 0 ? 0 : gx - r
  const x1 = gx + r > GRID ? GRID : gx + r
  const z0 = gz - r < 0 ? 0 : gz - r
  const z1 = gz + r > GRID ? GRID : gz + r
  let sum = 0
  for (let z = z0; z <= z1; z++) {
    const row = z * GRID_N
    for (let x = x0; x <= x1; x++) sum += grid[row + x]
  }
  return sum / ((x1 - x0 + 1) * (z1 - z0 + 1))
}

/**
 * Builds one chunk LOD mesh in world coordinates (meshes stay at the origin).
 * The vertex lattice has one extra ring on every side: those verts share the
 * edge position but drop by a LOD-scaled skirt depth, sealing cracks between
 * neighboring chunks at different LODs.
 *
 * LOD 0 reads the fine grid exactly (it must match the CPU collision
 * sampler); every coarser LOD reads through {@link filteredHeight} so its
 * surface is band-limited to its own vertex spacing. Vertex normals come
 * from central differences at the LOD's own stride on the same filtered
 * heights (per-pixel shading uses the baked normal texture instead, but the
 * attribute keeps three's pipeline seamless across chunk and LOD borders).
 */
function buildChunkGeometry(grid: Float32Array, ci: number, cj: number, lod: number): THREE.BufferGeometry {
  const step = LOD_STEP[lod]
  const segs = CHUNK_SEGS / step
  const vs = segs + 3 // verts per side, including the skirt ring
  const gx0 = ci * CHUNK_SEGS
  const gz0 = cj * CHUNK_SEGS
  const drop = CELL * step * 2 + 24
  const r = step === 1 ? 0 : step // box-filter radius: 0 at LOD 0, step beyond

  const positions = new Float32Array(vs * vs * 3)
  const normals = new Float32Array(vs * vs * 3)

  let w = 0
  for (let vz = -1; vz <= segs + 1; vz++) {
    const lz = vz < 0 ? 0 : vz > segs ? segs : vz
    const gz = gz0 + lz * step
    const z = -HALF + gz * CELL
    for (let vx = -1; vx <= segs + 1; vx++) {
      const lx = vx < 0 ? 0 : vx > segs ? segs : vx
      const gx = gx0 + lx * step
      const x = -HALF + gx * CELL
      const skirt = vx !== lx || vz !== lz || vx < 0 || vz < 0
      const y = filteredHeight(grid, gx, gz, r) - (skirt ? drop : 0)

      // Central-difference normal at this LOD's stride on filtered heights
      // (clamped at the world border).
      const d = step === 1 ? 1 : step
      const xm = gx - d < 0 ? 0 : gx - d
      const xp = gx + d > GRID ? GRID : gx + d
      const zm = gz - d < 0 ? 0 : gz - d
      const zp = gz + d > GRID ? GRID : gz + d
      const dhdx = (filteredHeight(grid, xp, gz, r) - filteredHeight(grid, xm, gz, r)) / ((xp - xm) * CELL)
      const dhdz = (filteredHeight(grid, gx, zp, r) - filteredHeight(grid, gx, zm, r)) / ((zp - zm) * CELL)
      const nl = 1 / Math.sqrt(dhdx * dhdx + 1 + dhdz * dhdz)

      positions[w] = x
      normals[w++] = -dhdx * nl
      positions[w] = y
      normals[w++] = nl
      positions[w] = z
      normals[w++] = -dhdz * nl
    }
  }

  // Index with the a→d diagonal so triangles match the CPU sampler exactly.
  const cells = vs - 1
  const indices = new Uint16Array(cells * cells * 6)
  let t = 0
  for (let r = 0; r < cells; r++) {
    for (let c = 0; c < cells; c++) {
      const a = r * vs + c
      const b = a + 1
      const cc = a + vs
      const dd = cc + 1
      indices[t++] = a
      indices[t++] = cc
      indices[t++] = dd
      indices[t++] = a
      indices[t++] = dd
      indices[t++] = b
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  geo.setIndex(new THREE.BufferAttribute(indices, 1))
  geo.computeBoundingSphere()
  return geo
}

// ---------------------------------------------------------------------------
// Surface material
// ---------------------------------------------------------------------------

/** GLSL noise toolkit + varyings + uniforms, prepended to the fragment shader. */
const FRAG_LIB = /* glsl */ `
varying vec3 vWorldPos;
uniform sampler2D avNormalTex;
uniform sampler2D avDetAlbedo;
uniform sampler2D avDetNormal;
// Hemisphere fill irradiance, filled by the surface block and consumed after
// lights_fragment_end so shadowed faces keep their albedo.
vec3 avHemi;

float avHash( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}
float avNoise( vec2 p ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  vec2 u = f * f * ( 3.0 - 2.0 * f );
  float a = avHash( i );
  float b = avHash( i + vec2( 1.0, 0.0 ) );
  float c = avHash( i + vec2( 0.0, 1.0 ) );
  float d = avHash( i + vec2( 1.0, 1.0 ) );
  return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );
}
float avFbm( vec2 p ) {
  float s = 0.0;
  float amp = 0.533;
  for ( int i = 0; i < 4; i ++ ) {
    s += amp * avNoise( p );
    p = p * 2.07 + 17.3;
    amp *= 0.5;
  }
  return s;
}
float avFbm3( vec2 p ) {
  float s = 0.0;
  float amp = 0.571;
  for ( int i = 0; i < 3; i ++ ) {
    s += amp * avNoise( p );
    p = p * 2.07 + 17.3;
    amp *= 0.5;
  }
  return s;
}
// Triplanar projections: sample each world plane, weight by the normal so
// steep faces read their own plane instead of a stretched top-down pattern.
float avTriFbm( vec3 p, vec3 w, float s ) {
  return w.x * avFbm3( p.zy * s ) + w.y * avFbm3( p.xz * s ) + w.z * avFbm3( p.xy * s );
}
float avTriNoise( vec3 p, vec3 w, float s ) {
  return w.x * avNoise( p.zy * s ) + w.y * avNoise( p.xz * s ) + w.z * avNoise( p.xy * s );
}
// One world-space triplanar detail cascade from the baked seamless maps.
// Samples albedo + bump-normal on all three world planes, selects the
// soil/grass/rock albedo channel per the material masks (albedo R/G/B,
// normal RG = soil, BA = rock), and accumulates a signed albedo modulation
// scalar plus a world-space normal perturbation. Per-plane gradients map to
// that plane's own axes, so cliff faces get unstretched vertical detail.
void avDetail( vec3 p, vec3 w, float s, float grassW, float rockW,
               float aStr, float nStr, inout float am, inout vec3 dn ) {
  vec2 uvx = p.zy * s;
  vec2 uvy = p.xz * s;
  vec2 uvz = p.xy * s;
  vec3 ax = texture2D( avDetAlbedo, uvx ).rgb;
  vec3 ay = texture2D( avDetAlbedo, uvy ).rgb;
  vec3 az = texture2D( avDetAlbedo, uvz ).rgb;
  vec3 aa = ax * w.x + ay * w.y + az * w.z;
  am += ( mix( mix( aa.r, aa.g, grassW ), aa.b, rockW ) - 0.5 ) * aStr;
  vec4 nx = texture2D( avDetNormal, uvx );
  vec4 ny = texture2D( avDetNormal, uvy );
  vec4 nz = texture2D( avDetNormal, uvz );
  vec2 gx = mix( nx.rg, nx.ba, rockW ) * 2.0 - 1.0;
  vec2 gy = mix( ny.rg, ny.ba, rockW ) * 2.0 - 1.0;
  vec2 gz = mix( nz.rg, nz.ba, rockW ) * 2.0 - 1.0;
  dn += ( vec3( 0.0, gx.y, gx.x ) * w.x + vec3( gy.x, 0.0, gy.y ) * w.y + vec3( gz.x, gz.y, 0.0 ) * w.z ) * nStr;
}
// Value noise with analytic derivatives (iq) — value in x, gradient in yz.
vec3 avNoised( vec2 p ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  vec2 u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
  vec2 du = 30.0 * f * f * ( f * ( f - 2.0 ) + 1.0 );
  float a = avHash( i );
  float b = avHash( i + vec2( 1.0, 0.0 ) );
  float c = avHash( i + vec2( 0.0, 1.0 ) );
  float d = avHash( i + vec2( 1.0, 1.0 ) );
  float k1 = b - a;
  float k2 = c - a;
  float k4 = a - b - c + d;
  float v = a + k1 * u.x + k2 * u.y + k4 * u.x * u.y;
  vec2 g = du * ( vec2( k1, k2 ) + k4 * u.yx );
  return vec3( v, g );
}
`

/**
 * Splat + detail-normal block. Replaces `normal_fragment_maps`, the point in
 * the standard shader where `diffuseColor`, `roughnessFactor` and the
 * view-space `normal` are all in scope and still mutable.
 *
 * Anti-repetition strategy (per the art-direction pass):
 * - The geometric normal comes from the baked heightfield normal texture, so
 *   shading is per-pixel smooth at every mesh LOD.
 * - Mid-scale breakup uses THREE octave scales cross-faded by km-scale macro
 *   noise, so no single cell size ever tiles across the world.
 * - Every cell-forming pattern is triplanar-projected.
 * - The procedural near grain lerps to flat albedo by ~210 m (the baked
 *   near cascade, which cannot shimmer, takes over from there), and the mid
 *   pattern only partially relaxes toward macro tint in the far field — the
 *   floor keeps distant country patterned instead of watercolor-flat.
 * - Sandstone stripes are slope-gated below ~50°, so steep faces and summits
 *   never carry wrap-around banding.
 */
const FRAG_SURFACE = /* glsl */ `
{
  vec2 avP = vWorldPos.xz;
  vec2 avUV = ( ( avP + vec2( ${HALF.toFixed(1)} ) ) * ${(1 / CELL).toFixed(8)} + 0.5 ) * ${(1 / GRID_N).toFixed(9)};
  vec3 avWN = normalize( texture2D( avNormalTex, avUV ).xyz * 2.0 - 1.0 );
  float avSlope = 1.0 - avWN.y;
  float avH = vWorldPos.y;
  float avCamD = distance( cameraPosition, vWorldPos );

  vec3 avTW = abs( avWN );
  avTW = avTW * avTW * avTW;
  avTW /= avTW.x + avTW.y + avTW.z;

  // Macro variation fields (km scale) steer every smaller pattern.
  float avMacro  = avFbm( avP * 0.00047 );
  float avMacro2 = avFbm( avP * 0.0021 + 31.9 );

  // Mid-scale breakup: three octave scales cross-faded by the macro fields.
  float avSel = clamp( avMacro * 1.9 - 0.45 + 0.5 * ( avMacro2 - 0.5 ), 0.0, 1.0 );
  float avMidA = avTriFbm( vWorldPos, avTW, 0.0062 );
  float avMidB = avTriFbm( vWorldPos, avTW, 0.019 );
  float avMidC = avTriFbm( vWorldPos, avTW, 0.051 );
  float avM2 = mix( avMidA, avMidB, smoothstep( 0.05, 0.60, avSel ) );
  avM2 = mix( avM2, avMidC, smoothstep( 0.50, 1.0, avSel ) * 0.65 );
  // Distant ground relaxes toward macro-tinted albedo, but only partially and
  // only far out: flattening from 500 m was what turned every aerial view
  // into watercolor blobs. The retained floor keeps mid-scale cells legible
  // to the horizon (their base wavelengths are tens of meters — no shimmer).
  float avFar = 1.0 - smoothstep( 900.0, 4200.0, avCamD );
  avM2 = mix( 0.5 + 0.55 * ( avMacro - 0.5 ), avM2, 0.45 + 0.55 * avFar );

  // Near-field grain, fully flat past ~210 m.
  float avNear = 1.0 - smoothstep( 80.0, 210.0, avCamD );
  float avMicro = mix( 0.5, avTriNoise( vWorldPos, avTW, 0.33 + 0.14 * avMacro2 ), avNear );
  float avGrain = mix( 0.5, avTriNoise( vWorldPos, avTW, 2.3 ), avNear * avNear );

  float avM1 = avFbm( avP * 0.0042 + vec2( 14.0 * avMacro, 3.1 ) );

  // Sandstone strata: height-banded, phase-warped so beds undulate.
  float avBandArg = avH * 0.42 + 6.0 * avFbm( avP * 0.006 ) + 3.0 * avMacro;
  float avBand = 0.5 + 0.5 * sin( avBandArg );
  float avCliff = smoothstep( 0.30, 0.55, avSlope + 0.12 * ( avM2 - 0.5 ) );

  // Palette (linear-light, tuned for ACES).
  vec3 ochre    = vec3( 0.335, 0.110, 0.048 );
  vec3 ochreHi  = vec3( 0.470, 0.200, 0.086 );
  vec3 spinifex = vec3( 0.430, 0.295, 0.105 );
  vec3 scrub    = vec3( 0.128, 0.152, 0.092 );
  vec3 sand     = vec3( 0.640, 0.535, 0.370 );
  vec3 stoneA   = vec3( 0.360, 0.200, 0.100 );
  vec3 stoneB   = vec3( 0.560, 0.375, 0.215 );
  vec3 rockHi   = vec3( 0.215, 0.170, 0.145 );

  vec3 avCol = mix( ochre, ochreHi, avM2 );

  float avGrass = smoothstep( 0.32, 0.62, avM1 )
                * ( 1.0 - smoothstep( 0.10, 0.28, avSlope ) )
                * smoothstep( 2.5, 9.0, avH )
                * ( 1.0 - smoothstep( 330.0, 560.0, avH ) );
  avCol = mix( avCol, spinifex * ( 0.78 + 0.5 * avM2 ), avGrass );

  float avScrub = smoothstep( 0.56, 0.78, avFbm( avP * 0.0058 + 41.7 ) )
                * ( 1.0 - smoothstep( 0.18, 0.42, avSlope ) )
                * smoothstep( 4.0, 22.0, avH );
  avCol = mix( avCol, scrub * ( 0.85 + 0.35 * avM2 ), avScrub * 0.9 );

  // Rock: stripes gate off above ~50° of slope, so dome flanks carry the
  // orange/grey bands while steep faces and points stay plain stone.
  float avRockM = max( avCliff, 0.85 * smoothstep( 0.10, 0.24, avSlope ) * smoothstep( 150.0, 320.0, avH ) );
  float avBandGate = 1.0 - smoothstep( 0.32, 0.44, avSlope );
  vec3 avStrata = mix( stoneA, stoneB, smoothstep( 0.2, 0.8, avBand ) ) * ( 0.85 + 0.3 * avM2 );
  vec3 avPlain = mix( rockHi, stoneA, 0.4 + 0.35 * avM2 );
  avCol = mix( avCol, mix( avPlain, avStrata, avBandGate ), avRockM );

  avCol = mix( avCol, rockHi * ( 0.8 + 0.45 * avM2 ),
               smoothstep( 560.0, 780.0, avH ) * ( 0.35 + 0.65 * avRockM ) );

  float avSand = ( 1.0 - smoothstep( 3.2, 7.0, avH ) ) * ( 1.0 - smoothstep( 0.14, 0.30, avSlope ) );
  avCol = mix( avCol, sand * ( 0.9 + 0.2 * avM2 ), avSand );

  // Wet sheen right at the waterline, teal darkening on the seabed.
  float avWet = avSand * ( 1.0 - smoothstep( 0.4, 1.8, avH ) );
  avCol = mix( avCol, avCol * vec3( 0.36, 0.50, 0.48 ), 1.0 - smoothstep( -8.0, 0.4, avH ) );

  // Baked triplanar detail cascades, three octave tiers: ~65 m structure
  // that NEVER distance-fades (the channels are centered on 0.5, so mip
  // minification converges to zero modulation on its own — no shimmer, no
  // pop), ~13 m breakup alive to ~9 km, and ~1.7 m grain alive to ~1.1 km so
  // the ground under a soaring bird still carries real texture. Tile scales
  // drift with the macro fields so the stamps never align across the world;
  // the rock channel rides the cliff mask, so escarpment faces carry cracked
  // bedded stone instead of smeared splats.
  float avGrassW = clamp( avGrass + avScrub, 0.0, 1.0 );
  float avDetStr = mix( 0.35, 1.0, smoothstep( -4.0, 1.0, avH ) ) * ( 1.0 - 0.6 * avSand );
  float avDAm = 0.0;
  vec3 avDN = vec3( 0.0 );
  avDetail( vWorldPos, avTW, 0.0154 * ( 0.88 + 0.24 * avMacro ), avGrassW, avRockM,
            0.55 * avDetStr, 0.7 * avDetStr, avDAm, avDN );
  float avDetFarF = 1.0 - smoothstep( 2500.0, 9000.0, avCamD );
  if ( avDetFarF > 0.002 ) {
    avDetail( vWorldPos, avTW, 0.0769 * ( 0.85 + 0.30 * avMacro ), avGrassW, avRockM,
              0.62 * avDetFarF * avDetStr, 1.0 * avDetFarF * avDetStr, avDAm, avDN );
    float avDetNearF = 1.0 - smoothstep( 150.0, 1100.0, avCamD );
    if ( avDetNearF > 0.002 ) {
      avDetail( vWorldPos, avTW, 0.588 * ( 0.82 + 0.36 * avMacro2 ), avGrassW, avRockM,
                0.85 * avDetNearF * avDetStr, 1.6 * avDetNearF * avDetStr, avDAm, avDN );
    }
  }
  avCol *= clamp( 1.0 + 1.7 * avDAm, 0.30, 1.85 );

  avCol *= 0.90 + 0.20 * avMacro;
  avCol *= 0.90 + 0.20 * avMicro;
  avCol *= 0.93 + 0.14 * avGrain;
  diffuseColor.rgb = avCol;

  roughnessFactor = clamp( roughnessFactor * ( 0.86 + 0.22 * avMicro ) - 0.06 * avGrass - 0.30 * avDAm, 0.05, 1.0 );
  roughnessFactor = mix( roughnessFactor, 0.32, avWet );

  // Procedural detail normals, distance-faded to kill shimmer AND pattern
  // repetition: the mid layer is gone past ~600 m so no single bump scale
  // ever stamps the whole view, and both layers' scales drift with the
  // macro fields so two hillsides never carry identical grain. Ground uses
  // xz-plane gradients; rock blends a vertical-plane sample plus slope-gated
  // strata ledges so faces read as bedded stone, not stretched noise.
  float avFadeHi = exp( -avCamD * 0.0045 );
  float avFadeLo = exp( -avCamD * 0.0035 );
  vec3 avD1 = avNoised( avP * ( 0.72 + 0.30 * avMacro2 ) );
  vec3 avD2 = avNoised( avP * ( 0.10 + 0.09 * avMacro ) + 47.1 );
  vec2 avGrad = avD1.yz * ( 0.42 * avFadeHi ) + avD2.yz * ( ( 0.28 + 0.36 * avMacro2 ) * avFadeLo );
  vec3 avD3 = avNoised( vec2( ( avP.x + avP.y ) * 0.5, avH ) * 0.6 );
  float avLedge = cos( avBandArg ) * 0.42 * avBandGate + avD3.z * 0.35;
  vec3 avPN = normalize( avWN + vec3( -avGrad.x, 0.0, -avGrad.y ) * ( 1.0 - 0.6 * avRockM ) );
  avPN = normalize( avPN - vec3( 0.0, ( avLedge * 0.6 * avFadeLo + avD3.z * 0.25 * avFadeHi ) * avRockM, 0.0 ) );
  // Baked detail-cascade bump on top: real 1.7 m / 13 m surface relief.
  avPN = normalize( avPN + avDN );
  normal = normalize( ( viewMatrix * vec4( avPN, 0.0 ) ).xyz );

  // Sky/ground hemisphere fill: shadowed faces keep albedo instead of
  // dropping to black under the single sun. BRDF_Lambert divides by pi, so
  // these irradiance values are pi-scaled to land near a 0.25-0.35 sky-color
  // ambient on the final albedo.
  avHemi = mix( vec3( 0.55, 0.38, 0.24 ), vec3( 0.95, 1.18, 1.55 ), avWN.y * 0.5 + 0.5 );
}
`

/** Hemisphere fill applied after all scene lights have accumulated. */
const FRAG_HEMI = /* glsl */ `
#include <lights_fragment_end>
reflectedLight.indirectDiffuse += avHemi * BRDF_Lambert( diffuseColor.rgb );
`

/**
 * Height-based exponential aerial perspective, replacing the stock
 * `fog_fragment` chunk (uniform FogExp2). Density follows
 * d(y) = fogDensity · exp(−(y − camY) / H) with H = 520 m, integrated
 * analytically along the view ray: valley floors sink into thicker haze
 * while peaks rise out of it, so distant relief keeps vertical separation
 * instead of compressing into one flat band. On a level ray the integral
 * reduces exactly to the stock FogExp2 result, so terrain fog stays
 * continuous with the Sky module's camera-altitude-modulated `fogDensity`
 * and with every other fogged material in the scene.
 *
 * The haze color warms near the horizon line on long rays and cools toward
 * sky blue on downward rays from altitude. The warm shift relaxes as fog
 * saturates so the far terrain limb still converges on the exact scene fog
 * color the sky dome closes on — no silhouette seam.
 */
const FRAG_FOG = /* glsl */ `
#if defined( USE_FOG ) && defined( FOG_EXP2 )
{
  vec3 avFV = vWorldPos - cameraPosition;
  float avFDist = max( length( avFV ), 1e-3 );
  float avFUp = avFV.y / avFDist;
  float avFu = clamp( avFV.y * ( 1.0 / 520.0 ), -8.0, 8.0 );
  float avFhf = abs( avFu ) < 1e-3 ? 1.0 : ( 1.0 - exp( -avFu ) ) / avFu;
  float avFe = fogDensity * avFDist * avFhf;
  float avFogA = 1.0 - exp( -avFe * avFe );
  float avFHor = exp( -abs( avFUp ) * 5.0 ) * smoothstep( 600.0, 3400.0, avFDist );
  float avFDown = clamp( -avFUp * 1.4, 0.0, 1.0 ) * smoothstep( 120.0, 900.0, cameraPosition.y );
  vec3 avFogC = fogColor * mix( vec3( 1.0 ), vec3( 1.17, 1.00, 0.78 ), avFHor * ( 1.0 - 0.6 * avFogA * avFogA ) );
  avFogC = mix( avFogC, fogColor * vec3( 0.82, 0.94, 1.22 ), avFDown * 0.62 * ( 1.0 - avFHor ) );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, avFogC, avFogA );
}
#else
#include <fog_fragment>
#endif
`

/**
 * Creates the single shared terrain material: a MeshStandardMaterial whose
 * albedo, roughness and normals are generated in-shader (geometric normals
 * come from the baked heightfield normal texture, surface relief from the
 * baked triplanar detail cascades). Standard lighting, shadow-receiving and
 * scene-environment chunks are untouched; the fog chunk is replaced with
 * height-based aerial perspective ({@link FRAG_FOG}).
 */
function createTerrainMaterial(normalTex: THREE.DataTexture, detail: DetailTextures): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1.0, metalness: 0.0 })
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.avNormalTex = { value: normalTex }
    shader.uniforms.avDetAlbedo = { value: detail.albedo }
    shader.uniforms.avDetNormal = { value: detail.normal }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWorldPos;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n  vWorldPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;',
      )
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + FRAG_LIB)
      .replace('#include <normal_fragment_maps>', FRAG_SURFACE)
      .replace('#include <lights_fragment_end>', FRAG_HEMI)
      .replace('#include <fog_fragment>', FRAG_FOG)
  }
  mat.customProgramCacheKey = () => 'aviary-terrain-v5'
  return mat
}

// ---------------------------------------------------------------------------
// Horizon ridgeline silhouettes
// ---------------------------------------------------------------------------

/**
 * Placement and tint recipe for the three silhouette rings that layer the
 * horizon beyond the world edge. Farther rings are taller (they must clear
 * the nearer ones over 40 km of view distance), lighter and closer to the
 * sky tone — classic aerial-perspective banding. `darken` scales the live
 * scene-fog color; `lift` lerps it toward the pale sky tone.
 */
const RIDGE_RINGS = [
  { radius: 9600, base: 90, amp: 620, freq: 4.2, seed: 3.7, darken: 0.84, lift: 0.06 },
  { radius: 13600, base: 210, amp: 980, freq: 3.1, seed: 41.2, darken: 0.94, lift: 0.45 },
  { radius: 19200, base: 360, amp: 1500, freq: 2.2, seed: 87.9, darken: 1.0, lift: 0.8 },
] as const

/** Angular segments per ridgeline ring. */
const RIDGE_SEGS = 768

/**
 * Builds one horizon silhouette ring: a vertical band centered on the origin
 * whose top edge is a periodic FBM ridgeline (noise sampled ON the unit
 * circle, so the seam closes exactly). The southern arc (+z, over the ocean)
 * drops to ~35 % amplitude — low headlands past the sea instead of an
 * implausible offshore mountain wall. The bottom edge sits below sea level
 * so no gap can open against the water.
 */
function buildRidgeRing(
  noise: (x: number, y: number) => number,
  radius: number,
  baseH: number,
  amp: number,
  freq: number,
  seed: number,
): THREE.BufferGeometry {
  const n = RIDGE_SEGS
  const positions = new Float32Array((n + 1) * 2 * 3)
  const indices = new Uint16Array(n * 6)
  let w = 0
  for (let i = 0; i <= n; i++) {
    const a = ((i % n) / n) * Math.PI * 2
    const cx = Math.cos(a)
    const sz = Math.sin(a)
    // Periodic FBM around the circle, normalized to 0..1.
    let s = 0
    let ampl = 0.5
    let norm = 0
    let f = freq
    for (let o = 0; o < 5; o++) {
      s += ampl * noise(cx * f + seed, sz * f + seed * 1.7)
      norm += ampl
      ampl *= 0.5
      f *= 2.03
    }
    let shape = 0.5 + (0.5 * s) / norm
    shape = shape * shape * (3 - 2 * shape) // rounded crests, deeper saddles
    const oceanAtt = 1 - 0.65 * sstep(0.1, 0.7, sz)
    const top = Math.max(25, (baseH + amp * shape * shape) * oceanAtt)
    const x = cx * radius
    const z = sz * radius
    positions[w++] = x
    positions[w++] = -60
    positions[w++] = z
    positions[w++] = x
    positions[w++] = top
    positions[w++] = z
  }
  let t = 0
  for (let i = 0; i < n; i++) {
    const b0 = i * 2
    const t0 = b0 + 1
    const b1 = b0 + 2
    const t1 = b0 + 3
    indices[t++] = b0
    indices[t++] = b1
    indices[t++] = t1
    indices[t++] = b0
    indices[t++] = t1
    indices[t++] = t0
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setIndex(new THREE.BufferAttribute(indices, 1))
  geo.computeBoundingSphere()
  return geo
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/** Per-chunk bookkeeping: its mesh, cached LOD geometries and LOD state. */
interface ChunkRecord {
  mesh: THREE.Mesh
  /** Lazily-built geometry per LOD level. */
  geoms: (THREE.BufferGeometry | null)[]
  /** Currently displayed LOD level. */
  lod: number
  /** Chunk center, world space (y = terrain height at the center). */
  cx: number
  cy: number
  cz: number
}

/**
 * The Terrain game module. Owns the baked fine heightfield, the exact
 * `getTerrainHeight` sampler, the heightfield-derived normal texture, 256
 * LOD-swapping chunk meshes and the shared splatted surface material.
 */
export class Terrain implements GameModule {
  readonly name = 'terrain'

  private grid!: Float32Array
  private chunks: ChunkRecord[] = []
  private material!: THREE.MeshStandardMaterial
  private normalTex!: THREE.DataTexture
  private detail!: DetailTextures
  /** The three horizon silhouette-ring materials, tinted live from scene fog. */
  private readonly ridgeMats: THREE.MeshBasicMaterial[] = []
  /** Scratch pale-sky tone for ring tinting — reused every frame. */
  private readonly ridgePale = new THREE.Color()
  /** Scratch camera position — reused every frame, never reallocated. */
  private readonly camPos = new THREE.Vector3()
  /** Squared LOD hand-over distances with enter/exit hysteresis margins. */
  private readonly lodEnterSq: number[] = LOD_DIST.map((d) => (d * 0.97) ** 2)
  private readonly lodExitSq: number[] = LOD_DIST.map((d) => (d * 1.03) ** 2)

  /** Bakes the heightfield, installs the sampler and spawns all chunk meshes. */
  init(ctx: GameContext): void {
    this.grid = generateHeightField()
    const sample = makeSampler(this.grid)
    ctx.getTerrainHeight = sample
    this.normalTex = bakeNormalTexture(this.grid)
    const aniso = Math.min(8, ctx.renderer.capabilities.getMaxAnisotropy())
    this.normalTex.anisotropy = aniso
    this.detail = bakeDetailTextures()
    this.detail.albedo.anisotropy = aniso
    this.detail.normal.anisotropy = aniso
    this.material = createTerrainMaterial(this.normalTex, this.detail)

    const group = new THREE.Group()
    group.name = 'terrain'
    group.matrixAutoUpdate = false

    for (let cj = 0; cj < CHUNKS; cj++) {
      for (let ci = 0; ci < CHUNKS; ci++) {
        const cx = -HALF + (ci + 0.5) * CHUNK_SIZE
        const cz = -HALF + (cj + 0.5) * CHUNK_SIZE
        const rec: ChunkRecord = {
          mesh: new THREE.Mesh(undefined, this.material),
          geoms: new Array<THREE.BufferGeometry | null>(LOD_COUNT).fill(null),
          lod: LOD_COUNT - 1,
          cx,
          cy: sample(cx, cz),
          cz,
        }
        rec.mesh.receiveShadow = true
        // Terrain must self-shadow: the NW range throws long shadows into the
        // valley at low sun angles. The depth-only pass is cheap (~400k tris).
        rec.mesh.castShadow = true
        rec.mesh.matrixAutoUpdate = false
        rec.mesh.geometry = this.geometry(rec, ci, cj, rec.lod)
        group.add(rec.mesh)
        this.chunks.push(rec)
      }
    }
    ctx.scene.add(group)

    // Horizon silhouette rings: three procedural ridgelines beyond the world
    // edge that layer the far clip into receding bands of aerial perspective.
    // Flat-shaded, fog-free (they ARE the fake fog layers), tinted per frame
    // from the live scene fog color in update().
    const ridgeNoise = createNoise2D(mulberry32(0x51dce5))
    const ridgeGroup = new THREE.Group()
    ridgeGroup.name = 'horizon-ridgelines'
    ridgeGroup.matrixAutoUpdate = false
    for (const r of RIDGE_RINGS) {
      const mat = new THREE.MeshBasicMaterial({ fog: false, side: THREE.DoubleSide })
      const mesh = new THREE.Mesh(buildRidgeRing(ridgeNoise, r.radius, r.base, r.amp, r.freq, r.seed), mat)
      mesh.matrixAutoUpdate = false
      mesh.frustumCulled = false // the camera is always inside the ring
      this.ridgeMats.push(mat)
      ridgeGroup.add(mesh)
    }
    ctx.scene.add(ridgeGroup)
    this.tintRidgelines(ctx)

    // Settle LOD around the spawn point so the first rendered frame is sharp.
    this.camPos.set(WORLD.spawn.x, WORLD.spawn.y, WORLD.spawn.z)
    this.refreshLod()
  }

  /**
   * Retargets chunk LODs at the camera and re-tints the horizon rings from
   * the live scene fog color. Zero allocations in steady state.
   */
  update(_dt: number, ctx: GameContext): void {
    ctx.camera.getWorldPosition(this.camPos)
    this.refreshLod()
    this.tintRidgelines(ctx)
  }

  /**
   * Derives each ring's flat color from the Sky-owned scene fog color:
   * nearest ring slightly darker and more saturated than the haze, farthest
   * lerped most of the way to a pale cool sky tone. Tracks the Sky module's
   * altitude-driven fog color shifts automatically, at zero allocation.
   */
  private tintRidgelines(ctx: GameContext): void {
    const fog = ctx.scene.fog
    if (!(fog instanceof THREE.FogExp2)) return
    this.ridgePale.copy(fog.color)
    this.ridgePale.r *= 1.06
    this.ridgePale.g *= 1.14
    this.ridgePale.b *= 1.30
    for (let i = 0; i < this.ridgeMats.length; i++) {
      const spec = RIDGE_RINGS[i]
      this.ridgeMats[i].color.copy(fog.color).multiplyScalar(spec.darken).lerp(this.ridgePale, spec.lift)
    }
  }

  /** Returns (building on first use) the geometry for a chunk LOD level. */
  private geometry(rec: ChunkRecord, ci: number, cj: number, lod: number): THREE.BufferGeometry {
    let geo = rec.geoms[lod]
    if (geo === null) {
      geo = buildChunkGeometry(this.grid, ci, cj, lod)
      rec.geoms[lod] = geo
    }
    return geo
  }

  /** Walks every chunk and swaps LOD when the camera crosses a hysteresis band. */
  private refreshLod(): void {
    const px = this.camPos.x
    const py = this.camPos.y
    const pz = this.camPos.z
    for (let i = 0; i < this.chunks.length; i++) {
      const rec = this.chunks[i]
      const dx = rec.cx - px
      const dy = rec.cy - py
      const dz = rec.cz - pz
      const d2 = dx * dx + dy * dy + dz * dz
      let lod = rec.lod
      while (lod < LOD_COUNT - 1 && d2 > this.lodExitSq[lod]) lod++
      while (lod > 0 && d2 < this.lodEnterSq[lod - 1]) lod--
      if (lod !== rec.lod) {
        rec.lod = lod
        rec.mesh.geometry = this.geometry(rec, i % CHUNKS, (i / CHUNKS) | 0, lod)
      }
    }
  }
}
