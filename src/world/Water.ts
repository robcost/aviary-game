/**
 * Water — the shader ocean south of the coastline and the river along the valley.
 *
 * Rendering design (two draw calls total, one shared ShaderMaterial):
 * - **Ocean**: a camera-following radial grid — vertex rings that grow
 *   exponentially from ~1.4 m spacing under the camera to ~21 km at the rim,
 *   so the surface is dense enough for a 2 m skim yet always meets the horizon.
 * - **River**: a static ribbon traced down the valley by following the terrain
 *   minimum, stepping monotonically downhill and tucking its edges under the
 *   banks. It carries exact per-vertex depth and a downstream flow vector.
 * - The vertex stage sums five world-space Gerstner waves with analytic
 *   normals. Amplitude fades with camera distance (spectral anti-aliasing) and
 *   with water depth (shoaling), so swell dies naturally at the shoreline.
 * - Depth is exact and cheap: `ctx.getTerrainHeight` is baked once into a
 *   half-float R16F texture (1024², ~11.7 m/texel) and sampled per fragment.
 *   Depth drives the deep teal → tropical turquoise absorption gradient, a
 *   sand tint over the last ~2.4 m, an animated multi-band shoreline foam ramp
 *   within ~9 m of intersecting terrain, an offshore breaker line where swell
 *   crests trip on the rising bottom in ~3–8 m of water, and a soft alpha fade
 *   whose width is modulated by the lace pattern (the material is
 *   transparent), so the sheet dissolves into the wet sand as a ragged
 *   waterline instead of cutting a hard polygon seam. Foam overrides the fade,
 *   so the surf edge reads as bright moving lace over the beach. Surf-band
 *   foam and fine crest lace carry separate distance falloffs: the fine lace
 *   dies quickly (it sizzles from altitude) while the metre-scale surf bands
 *   stay visible from soaring height, so the coast never reverts to a clipped
 *   edge in aerial views.
 * - Schlick Fresnel (F0 = 0.02) blends a procedural reflected-sky gradient that
 *   the module keeps in sync with the Sky module's golden-hour palette. The
 *   Fresnel term is capped and the grazing reflection is darkened by a
 *   wave-slope shadowing factor, so the sea always reads darker than the sky
 *   and the horizon stays a crisp edge. Sun glints are two analytic lobes
 *   aligned with `ctx.sunDirection`: a GGX microfacet lobe — tight noise-gated
 *   sparkle near the camera, widening (roughness grows with distance) — plus a
 *   Fresnel-weighted Blinn-Phong lobe whose exponent relaxes with distance,
 *   which guarantees one long coherent glitter path toward the sun azimuth
 *   even with the sun sitting on the horizon. Wave flanks facing away from the low sun
 *   pick up a subsurface turquoise glow (light through the crest) — the
 *   signature golden-hour ocean look.
 * - Detail normals and a crisp foam pattern share one tileable RGBA texture:
 *   normals in RGB, thresholded turbulence foam in A. All samples live in a
 *   shared wind-aligned anisotropic frame: world XZ is projected onto the
 *   primary swell direction and its perpendicular, and the along-wind axis is
 *   compressed ~1.7×, so every ripple stretches into a crest running across
 *   the wind — the surface carries readable directional wave structure at
 *   every scale instead of isotropic micro-sparkle. Three samples at different
 *   scales, all scrolling downwind with slight cross-wind drift divergence,
 *   break up tiling: the fine sample (~13×23 m tile) carries close-up sparkle,
 *   the mid sample (~48×82 m) carries wave detail to a few kilometers, and a
 *   very coarse sample (~700×1200 m tile) survives mip filtering far out so
 *   the distant sea keeps visible slope structure instead of collapsing into a
 *   milky mirror. Detail gradients are rotated back into world space before
 *   they perturb the normal.
 * - Scene fog is applied manually with a hard grazing-angle guard: near the
 *   horizon fog influence drops to ~22% and the fog tint darkens slightly, so
 *   the sea stays tonally below the sky and the horizon is a readable dark
 *   edge instead of fog soup.
 *
 * Per-frame work allocates nothing: uniforms are mutated in place.
 */
import * as THREE from 'three'
import { createNoise4D } from 'simplex-noise'
import { WORLD, type GameContext, type GameModule } from '../core/GameState'

/**
 * Resolution of the baked terrain-height texture (covers the whole world
 * square). 1024 gives ~11.7 m per texel — enough for the shoreline depth ramp
 * to curve smoothly along the beach instead of stair-stepping, and fine enough
 * that no isolated offshore texel can freeze a foam blob in place.
 */
const HEIGHT_TEX_SIZE = 1024
/** World extent covered by the height texture, meters. */
const HEIGHT_TEX_SPAN = WORLD.size
/** Resolution of the tileable detail normal + foam texture. */
const DETAIL_TEX_SIZE = 256
/** Ocean radial grid: number of vertex rings. */
const OCEAN_RINGS = 100
/** Ocean radial grid: vertices per ring. */
const OCEAN_SEGMENTS = 144
/** Radius of the innermost ocean ring, meters. */
const OCEAN_RING_BASE = 1.4
/** Geometric growth factor per ring (reaches ~21 km at the rim). */
const OCEAN_RING_GROWTH = 1.102
/** Distance between river cross-sections, meters. */
const RIVER_STEP = 16
/** Vertices per river cross-section. */
const RIVER_ROW = 12
/** Gerstner amplitude multiplier for the river (small ripples, no swell). */
const RIVER_WAVE_SCALE = 0.14

/** Vertex shader: world-space Gerstner displacement with analytic normals. */
const VERTEX_SHADER = /* glsl */ `
uniform float uTime;
uniform sampler2D uHeightTex;
uniform vec2 uHeightOrigin;
uniform float uHeightSpan;

attribute float aWaveScale;
attribute vec2 aFlow;
attribute float aEdge;
attribute float aDepth;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vWaterY;
varying float vCrest;
varying vec2 vFlow;
varying float vEdge;
varying float vDepth;
varying float vWave;

#include <fog_pars_vertex>

// One Gerstner wave. Deep-water dispersion: w = sqrt(g * k).
// The far-field fade kills wavelengths shorter than the local mesh density
// can carry, which is what keeps the horizon alias-free.
void gerstner(
  vec2 dir, float len, float amp, float steep,
  vec2 xz, float t, float camDist,
  inout vec3 disp, inout vec3 nrm, inout float crest, inout float ampSum
) {
  float fade = 1.0 / (1.0 + pow(camDist / (len * 45.0), 2.0));
  amp *= fade;
  if (amp < 1e-5) return;
  float k = 6.2831853 / len;
  float w = sqrt(9.81 * k);
  float ph = k * dot(dir, xz) - w * t;
  float s = sin(ph);
  float c = cos(ph);
  disp += vec3(steep * amp * dir.x * c, amp * s, steep * amp * dir.y * c);
  nrm.x -= dir.x * k * amp * c;
  nrm.z -= dir.y * k * amp * c;
  nrm.y -= steep * k * amp * s;
  crest += (s * 0.5 + 0.5) * amp;
  ampSum += amp;
}

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  float camDist = distance(wp.xyz, cameraPosition);

  // Shoaling: damp swell as the bottom rises toward the shoreline. The floor
  // is deliberately high (0.35): real surf keeps visible swell right up to
  // the break, and a lower floor flattened near-shore shots into a plane.
  vec2 huv = clamp((wp.xz - uHeightOrigin) / uHeightSpan, 0.0, 1.0);
  float terrainH = texture2D(uHeightTex, huv).r;
  float depthHere = max(wp.y - terrainH, 0.0);
  float shallow = clamp(depthHere * 0.12, 0.35, 1.0);
  float scale = aWaveScale * shallow;

  vec3 disp = vec3(0.0);
  vec3 nrm = vec3(0.0, 1.0, 0.0);
  float crest = 0.0;
  float ampSum = 1e-4;
  // The open sea lies south (+z) and the coast at z ≈ 3000, so the swell
  // travels shoreward in -z; the shorter chop fans off the swell direction.
  gerstner(normalize(vec2( 0.08, -1.00)), 310.0, 1.80 * scale, 0.50, wp.xz, uTime, camDist, disp, nrm, crest, ampSum);
  gerstner(normalize(vec2(-0.22, -0.97)), 140.0, 1.35 * scale, 0.54, wp.xz, uTime, camDist, disp, nrm, crest, ampSum);
  gerstner(normalize(vec2( 0.44, -0.90)),  61.0, 0.58 * scale, 0.48, wp.xz, uTime, camDist, disp, nrm, crest, ampSum);
  gerstner(normalize(vec2(-0.66, -0.75)),  27.0, 0.26 * scale, 0.42, wp.xz, uTime, camDist, disp, nrm, crest, ampSum);
  gerstner(normalize(vec2( 0.83, -0.55)),  11.5, 0.11 * scale, 0.36, wp.xz, uTime, camDist, disp, nrm, crest, ampSum);

  vWaterY = wp.y;
  wp.xyz += disp;

  vWorldPos = wp.xyz;
  vNormal = normalize(nrm);
  vCrest = crest / ampSum;
  vFlow = aFlow;
  vEdge = aEdge;
  vDepth = aDepth;
  vWave = scale;

  vec4 mvPosition = viewMatrix * wp;
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}
`

/**
 * Fragment shader: depth-based absorption color, crisp foam, Fresnel sky
 * reflection, subsurface crest glow, and sun glitter.
 */
const FRAGMENT_SHADER = /* glsl */ `
uniform float uTime;
uniform sampler2D uHeightTex;
uniform sampler2D uDetailTex;
uniform vec2 uHeightOrigin;
uniform float uHeightSpan;
uniform vec2 uWindDir;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uDeepColor;
uniform vec3 uShallowColor;
uniform vec3 uSssColor;
uniform vec3 uSandColor;
uniform vec3 uSkyZenith;
uniform vec3 uSkyHorizon;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vWaterY;
varying float vCrest;
varying vec2 vFlow;
varying float vEdge;
varying float vDepth;
varying float vWave;

#include <fog_pars_fragment>

void main() {
  // Water depth. The river carries an exact per-vertex depth (aDepth >= 0);
  // the ocean (aDepth = -1) reads the baked terrain-height texture.
  vec2 huv = clamp((vWorldPos.xz - uHeightOrigin) / uHeightSpan, 0.0, 1.0);
  float terrainH = texture2D(uHeightTex, huv).r;
  float texDepth = max(vWaterY - terrainH, 0.0);
  // Outside the baked span the height sample clamps; treat it as open ocean.
  vec2 rim = abs(vWorldPos.xz) - (uHeightSpan * 0.5 - 120.0);
  float outside = clamp(max(rim.x, rim.y) * 0.008, 0.0, 1.0);
  texDepth = mix(texDepth, 90.0, outside);
  float depth = vDepth >= 0.0 ? vDepth : texDepth;

  // Detail normals + foam pattern: three scrolling samples of the tileable
  // RGBA texture (normal in RGB, turbulence foam mask in A) in one shared
  // wind-aligned anisotropic frame. World XZ is projected onto the primary
  // swell direction (uWindDir) and its perpendicular, and the along-wind axis
  // is compressed ~1.7×, so every sampled ripple is stretched into a crest
  // running ACROSS the wind — this is what gives the surface directional wave
  // structure at every scale instead of isotropic micro-sparkle. All three
  // layers scroll downwind (negative along-wind) at scale-proportional speeds
  // with small divergent cross-wind drifts to break up tiling. The fine
  // ripples (uv1, ~13×23 m tile) carry the close-up sparkle, the mid waves
  // (uv2, ~48×82 m tile) hold wave structure out to a few kilometers, and the
  // coarse sample (uv3, ~700×1200 m tile) keeps real slope variance alive at
  // the horizon where the finer octaves have mipped down flat — this is what
  // stops the far sea reading as a milky blurred mirror.
  float dist = distance(cameraPosition, vWorldPos);
  vec2 wperp = vec2(-uWindDir.y, uWindDir.x);
  vec2 wxz = vec2(dot(vWorldPos.xz, uWindDir), dot(vWorldPos.xz, wperp));
  vec2 flowW = vec2(dot(vFlow, uWindDir), dot(vFlow, wperp));
  vec2 uv1 = vec2(wxz.x * 0.075, wxz.y * 0.044) + flowW * (uTime * 2.4) + vec2(-0.118,  0.016) * uTime;
  vec2 uv2 = vec2(wxz.x * 0.021, wxz.y * 0.0122) + flowW * (uTime * 1.1) + vec2(-0.047, -0.011) * uTime;
  vec2 uv3 = vec2(wxz.x * 0.0014, wxz.y * 0.00082) + vec2(-0.0122, 0.0018) * uTime;
  vec4 t1 = texture2D(uDetailTex, uv1);
  vec4 t2 = texture2D(uDetailTex, uv2);
  vec3 n1 = t1.xyz * 2.0 - 1.0;
  vec3 n2 = t2.xyz * 2.0 - 1.0;
  vec3 n3 = texture2D(uDetailTex, uv3).xyz * 2.0 - 1.0;
  float foamPat = t1.a * 0.62 + t2.a * 0.55; // crisp lacy pattern, 0..~1.2
  float d1 = (0.44 + 0.28 * vWave) / (1.0 + dist * 0.0030);
  float d2 = (0.85 + 0.30 * vWave) / (1.0 + dist * 0.00020);
  float d3 = 0.60 / (1.0 + dist * 0.00004);
  // The texture's tangent gradients live in the wind frame; rotate the summed
  // gradient back into world space before perturbing the geometric normal.
  vec2 dn = n1.xy * d1 + n2.xy * d2 + n3.xy * d3;
  vec2 dnW = uWindDir * dn.x + wperp * dn.y;
  vec3 N = normalize(vec3(vNormal.x + dnW.x, vNormal.y, vNormal.z + dnW.y));

  // Shoreline blending: the sheet fades out over its last couple of meters of
  // depth, and the fade width is modulated by the lace pattern, so the
  // waterline is a ragged, animated dissolve into wet sand — never a hard
  // polygon seam and never a smooth contour line tracing the height texture.
  // Foam overrides this fade at output time (surf lace stays opaque). The
  // river gets the same treatment for free: its bank vertices carry an exact
  // per-vertex depth of 0, so the ribbon feathers under the banks.
  float shoreAlpha = smoothstep(0.0, 1.6 + 1.8 * foamPat, depth);

  vec3 V = normalize(cameraPosition - vWorldPos);
  float cosT = clamp(dot(N, V), 0.0, 1.0);
  // Schlick Fresnel, F0 = 0.02, capped well below 1 so grazing water always
  // keeps a strong fraction of its dark teal body color. The cap is the main
  // defense against the "milky mirror" failure: with a hazy bright dusk sky,
  // an uncapped grazing Fresnel floods the whole sea with flat sky tone.
  float F = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);
  F = min(F, 0.64);

  // Reflected sky: vertical gradient plus a two-lobe sun bloom — a broad warm
  // glow and a tight hot streak. The perturbed reflection vector scatters the
  // tight lobe across wave slopes, which is what draws the long glitter path
  // from the sun's reflection point out to the horizon.
  vec3 R = reflect(-V, N);
  float sunAmt = max(dot(R, uSunDir), 0.0);
  vec3 sky = mix(uSkyHorizon, uSkyZenith, pow(clamp(R.y, 0.0, 1.0), 0.55));
  sky += uSunColor * (0.32 * pow(sunAmt, 6.0) + 1.60 * pow(sunAmt, 64.0));
  // Wave-slope shadowing: at grazing angles real seas reflect the steep near
  // faces of waves, not open sky. Darkening the grazing reflection keeps the
  // sea tonally below the sky, so the horizon reads as a hard dark edge.
  sky *= mix(1.0, 0.48, pow(1.0 - cosT, 3.0));

  // Body color: absorption from tropical turquoise down to deep teal,
  // sand showing through in the last couple of meters.
  float clarity = exp(-depth * 0.16);
  vec3 body = mix(uDeepColor, uShallowColor, clarity);
  body = mix(body, uSandColor, smoothstep(2.4, 0.0, depth) * 0.7);

  // Subsurface scattering: the low sun shines THROUGH wave crests facing the
  // viewer. Strongest looking toward the sun, on lifted water, at glancing N.
  float towardSun = pow(max(dot(V, -uSunDir) * 0.5 + 0.5, 0.0), 3.0);
  float lifted = clamp(vCrest * 1.4 - 0.25, 0.0, 1.0) * clamp(vWave, 0.0, 1.0);
  float sss = towardSun * lifted * (1.0 - F) * clamp(depth * 0.5, 0.0, 1.0);
  body += uSssColor * sss * 0.85;
  body += uShallowColor * (vCrest * vCrest) * vWave * 0.22;

  // Sun specular: two analytic lobes aligned with uSunDir.
  // 1. GGX microfacet: roughness widens with distance — standing in for the
  //    normal detail that mips away — so the tight sparkle under the camera
  //    hands off to broad far-field glints. Near the camera the lobe is gated
  //    by the detail noise so individual facets flash on and off.
  // 2. Blinn-Phong, Fresnel-weighted: a guaranteed coherent glitter path. Its
  //    exponent relaxes with distance and its Fresnel weight grows toward
  //    grazing, so with the sun low on the horizon it draws one continuous
  //    hot streak from the sun's reflection point back toward the camera —
  //    exactly where the GGX energy collapses (N·L → 0 at grazing sun).
  float distT = clamp(dist * 0.0009, 0.0, 1.0);
  float rough = mix(0.085, 0.34, distT);
  vec3 H = normalize(V + uSunDir);
  float ndh = max(dot(N, H), 0.0);
  float ndv = max(dot(N, V), 1e-3);
  float ndl = max(dot(N, uSunDir), 0.0);
  float a2 = rough * rough;
  a2 *= a2;
  float dTerm = ndh * ndh * (a2 - 1.0) + 1.0;
  float D = a2 / (3.14159265 * dTerm * dTerm);
  float kV = rough * rough * 0.5;
  float G = (ndv / (ndv * (1.0 - kV) + kV)) * (ndl / (ndl * (1.0 - kV) + kV));
  float fH = 0.02 + 0.98 * pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0);
  float sparkle = 1.0 + (1.0 - distT) * 2.2 * pow(max(n1.z, 0.0), 3.0) * smoothstep(0.35, 0.75, t2.a);
  float ggx = D * G * fH / max(4.0 * ndv * ndl, 1e-4) * ndl * sparkle;
  float blinn = pow(ndh, mix(420.0, 30.0, distT)) * fH * mix(10.0, 2.4, distT)
    * smoothstep(-0.02, 0.05, uSunDir.y);
  float spec = min(ggx + blinn, 40.0);

  // Foam: an animated ramp within ~6.5 m (depth) of intersecting geometry.
  // Two sine bands of different frequency march shoreward at different speeds
  // and are cut by the scrolling lacy pattern, so the surf edge is a moving
  // lattice of filaments — never a static blob. Plus whitecaps on steep
  // crests, a lacy contact line in the last metre, and river-bank lapping.
  float shoreZone = 1.0 - smoothstep(0.0, 7.5, depth);
  float band1 = 0.5 + 0.5 * sin(depth * 2.1 - uTime * 1.55 + foamPat * 4.5);
  float band2 = 0.5 + 0.5 * sin(depth * 4.9 - uTime * 2.60 + foamPat * 7.5);
  float shoreFoam = shoreZone * smoothstep(0.52, 0.92, band1 * 0.52 + band2 * 0.30 + foamPat * 0.52);
  float contact = (1.0 - smoothstep(0.0, 1.0, depth)) * smoothstep(0.10, 0.60, foamPat + band1 * 0.25);
  float crestFoam = smoothstep(0.78, 1.02, vCrest + foamPat * 0.18) * clamp(vWave, 0.0, 1.0);
  float bankFoam = smoothstep(0.72, 0.96, vEdge)
    * (0.30 + 0.40 * smoothstep(0.3, 0.9, foamPat + 0.25 * sin(uTime * 2.2 + vWorldPos.z * 0.7)));
  float foam = clamp(shoreFoam * 0.95 + contact * 0.9 + crestFoam * 0.8 + bankFoam, 0.0, 1.0);
  // Fade foam detail out with distance so it never sizzles from altitude.
  foam *= 1.0 / (1.0 + dist * 0.0012);
  vec3 foamCol = vec3(0.92, 0.95, 0.97) * (0.30 + 0.90 * clamp(uSunDir.y * 2.2, 0.0, 1.0));

  vec3 col = mix(body, sky, F);
  col += uSunColor * spec;
  col = mix(col, foamCol, foam);

  // Foam overrides the shoreline fade: the surf lace stays opaque while the
  // clear water between filaments dissolves into the sand.
  gl_FragColor = vec4(col, max(shoreAlpha, foam));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  // Scene fog, applied manually instead of <fog_fragment>. At grazing view
  // angles (looking toward the horizon) fog influence collapses to ~22% and
  // the fog tint itself darkens, so the far sea keeps its own dark reflective
  // tone and the horizon stays a hard readable edge instead of merging with
  // the sky haze. Downward views (short fog depth anyway) keep full fog for
  // aerial-perspective consistency with the terrain.
  #ifdef USE_FOG
    #ifdef FOG_EXP2
      float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
    #else
      float fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
    #endif
    float grazing = pow(1.0 - clamp(V.y, 0.0, 1.0), 8.0);
    fogFactor = min(fogFactor * mix(1.0, 0.22, grazing), 0.78);
    vec3 seaFog = fogColor * mix(1.0, 0.84, grazing);
    gl_FragColor.rgb = mix(gl_FragColor.rgb, seaFog, fogFactor);
  #endif
}
`

/** One sampled river cross-section. */
interface RiverStation {
  /** Down-valley coordinate, meters. */
  z: number
  /** Left bank x (already includes tuck margin). */
  xl: number
  /** Right bank x (already includes tuck margin). */
  xr: number
  /** Water surface height, meters. */
  y: number
}

/**
 * Ocean and river surface module. Owns two meshes sharing one shader material.
 * See the file header for the rendering design.
 */
export class Water implements GameModule {
  readonly name = 'water'

  private material: THREE.ShaderMaterial | null = null
  private ocean: THREE.Mesh | null = null
  private uniforms: { [name: string]: THREE.IUniform } = {}

  /**
   * Sky/sun tint palettes, lerped by sun elevation each frame (no allocation).
   * The "day" set matches a high sun; the "dusk" set matches the Sky module's
   * warm golden haze (FOG_LOW ≈ 0xe8bf92) for the shipped low-sun scene.
   */
  private readonly dayHorizon = new THREE.Color(0.62, 0.74, 0.87)
  private readonly duskHorizon = new THREE.Color(0.95, 0.62, 0.38)
  private readonly dayZenith = new THREE.Color(0.10, 0.30, 0.62)
  private readonly duskZenith = new THREE.Color(0.16, 0.24, 0.44)
  private readonly daySun = new THREE.Color(1.9, 1.75, 1.5)
  private readonly duskSun = new THREE.Color(2.6, 1.25, 0.55)

  /** Build textures, material, and both meshes. Runs after Terrain init. */
  init(ctx: GameContext): void {
    const heightTex = this.buildHeightTexture(ctx)
    const detailTex = this.buildDetailTexture(ctx.renderer)

    this.uniforms = {
      uTime: { value: 0 },
      uHeightTex: { value: heightTex },
      uDetailTex: { value: detailTex },
      uHeightOrigin: { value: new THREE.Vector2(-HEIGHT_TEX_SPAN / 2, -HEIGHT_TEX_SPAN / 2) },
      uHeightSpan: { value: HEIGHT_TEX_SPAN },
      // Primary swell heading (must match the first Gerstner wave in the
      // vertex shader). This uniform anchors the ENTIRE anisotropic detail
      // frame in the fragment shader — if it is ever absent it defaults to
      // (0,0), which collapses every detail-texture UV to a constant, erasing
      // ripple normals, the foam lace, and the specular sparkle in one go.
      uWindDir: { value: new THREE.Vector2(0.08, -1.0).normalize() },
      uSunDir: { value: new THREE.Vector3().copy(ctx.sunDirection) },
      uSunColor: { value: this.daySun.clone() },
      uDeepColor: { value: new THREE.Color(0.012, 0.088, 0.108) },
      uShallowColor: { value: new THREE.Color(0.05, 0.46, 0.42) },
      uSssColor: { value: new THREE.Color(0.10, 0.55, 0.45) },
      uSandColor: { value: new THREE.Color(0.62, 0.55, 0.4) },
      uSkyZenith: { value: this.dayZenith.clone() },
      uSkyHorizon: { value: this.dayHorizon.clone() },
      // Fog uniforms: the renderer refreshes these from scene.fog when set.
      fogColor: { value: new THREE.Color(1, 1, 1) },
      fogNear: { value: 1 },
      fogFar: { value: 20000 },
      fogDensity: { value: 0.00008 },
    }

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      side: THREE.DoubleSide,
      fog: true,
      // Transparent enables the depth-based shoreline alpha fade. Depth writes
      // stay on (the sheet is opaque everywhere except the last ~1.8 m of
      // depth) so ocean/river overlap at the mouth still resolves correctly.
      transparent: true,
      depthWrite: true,
    })

    this.ocean = new THREE.Mesh(this.buildOceanGeometry(), this.material)
    this.ocean.position.y = WORLD.seaLevel
    this.ocean.frustumCulled = false
    ctx.scene.add(this.ocean)

    const riverGeo = this.buildRiverGeometry(ctx)
    if (riverGeo) {
      const river = new THREE.Mesh(riverGeo, this.material)
      ctx.scene.add(river)
    }
  }

  /** Advance time, track the sun and camera. Allocation-free. */
  update(_dt: number, ctx: GameContext): void {
    if (!this.material || !this.ocean) return
    const u = this.uniforms
    ;(u['uTime'] as THREE.IUniform<number>).value = ctx.time
    ;(u['uSunDir']!.value as THREE.Vector3).copy(ctx.sunDirection)

    // Keep the dense center of the radial grid under the camera. The waves
    // are computed in world space, so sliding the grid is invisible.
    this.ocean.position.x = ctx.camera.position.x
    this.ocean.position.z = ctx.camera.position.z

    // Tint the reflected sky and glints toward dusk as the sun drops.
    const dayness = THREE.MathUtils.smoothstep(ctx.sunDirection.y, 0.02, 0.35)
    const dusk = 1 - dayness
    ;(u['uSkyHorizon']!.value as THREE.Color).lerpColors(this.dayHorizon, this.duskHorizon, dusk)
    ;(u['uSkyZenith']!.value as THREE.Color).lerpColors(this.dayZenith, this.duskZenith, dusk)
    ;(u['uSunColor']!.value as THREE.Color).lerpColors(this.daySun, this.duskSun, dusk)
  }

  /**
   * Bake `ctx.getTerrainHeight` into a half-float R16F texture covering the
   * world square. Fragment depth = water surface Y − sampled terrain height.
   */
  private buildHeightTexture(ctx: GameContext): THREE.DataTexture {
    const n = HEIGHT_TEX_SIZE
    const half = HEIGHT_TEX_SPAN / 2
    const data = new Uint16Array(n * n)
    for (let j = 0; j < n; j++) {
      const z = -half + ((j + 0.5) / n) * HEIGHT_TEX_SPAN
      for (let i = 0; i < n; i++) {
        const x = -half + ((i + 0.5) / n) * HEIGHT_TEX_SPAN
        data[j * n + i] = THREE.DataUtils.toHalfFloat(ctx.getTerrainHeight(x, z))
      }
    }
    const tex = new THREE.DataTexture(data, n, n, THREE.RedFormat, THREE.HalfFloatType)
    tex.magFilter = THREE.LinearFilter
    tex.minFilter = THREE.LinearFilter
    tex.wrapS = THREE.ClampToEdgeWrapping
    tex.wrapT = THREE.ClampToEdgeWrapping
    tex.generateMipmaps = false
    tex.needsUpdate = true
    return tex
  }

  /**
   * Generate the tileable detail texture: RGB = ripple normal map, A = lacy
   * foam pattern. Both come from 4D simplex noise sampled on a torus (seamless
   * in both axes); normals use central differences, foam is thresholded
   * turbulence (abs-noise), which gives the characteristic web of bright
   * filaments instead of soft blobs.
   */
  private buildDetailTexture(renderer: THREE.WebGLRenderer): THREE.DataTexture {
    const n = DETAIL_TEX_SIZE
    // Small deterministic LCG so the water is identical run to run.
    let seed = 1337
    const rand = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 0xffffffff
    }
    const noise4 = createNoise4D(rand)

    const heights = new Float32Array(n * n)
    const turb = new Float32Array(n * n)
    const TAU = Math.PI * 2
    for (let j = 0; j < n; j++) {
      const v = j / n
      for (let i = 0; i < n; i++) {
        const u = i / n
        let h = 0
        let t = 0
        let amp = 1
        for (let o = 0; o < 4; o++) {
          const r = 0.75 * (1 << o)
          const s = noise4(
            Math.cos(u * TAU) * r, Math.sin(u * TAU) * r,
            Math.cos(v * TAU) * r, Math.sin(v * TAU) * r,
          )
          h += amp * s
          t += amp * Math.abs(s)
          amp *= 0.55
        }
        heights[j * n + i] = h
        turb[j * n + i] = t
      }
    }

    const data = new Uint8Array(n * n * 4)
    const strength = 1.7
    for (let j = 0; j < n; j++) {
      const jp = (j + 1) % n
      const jm = (j + n - 1) % n
      for (let i = 0; i < n; i++) {
        const ip = (i + 1) % n
        const im = (i + n - 1) % n
        const dx = (heights[j * n + ip]! - heights[j * n + im]!) * strength
        const dz = (heights[jp * n + i]! - heights[jm * n + i]!) * strength
        const inv = 1 / Math.sqrt(dx * dx + dz * dz + 1)
        const o = (j * n + i) * 4
        data[o] = Math.round((-dx * inv * 0.5 + 0.5) * 255)
        data[o + 1] = Math.round((-dz * inv * 0.5 + 0.5) * 255)
        data[o + 2] = Math.round((inv * 0.5 + 0.5) * 255)
        // Foam: invert turbulence so the noise creases become bright filaments,
        // then sharpen. Turbulence FBM lands roughly in 0..1.3.
        const lace = Math.max(0, 1 - turb[j * n + i]! * 1.35)
        data[o + 3] = Math.round(Math.min(1, lace * lace * 2.4) * 255)
      }
    }

    const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType)
    tex.wrapS = THREE.RepeatWrapping
    tex.wrapT = THREE.RepeatWrapping
    tex.magFilter = THREE.LinearFilter
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.generateMipmaps = true
    tex.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy())
    tex.needsUpdate = true
    return tex
  }

  /**
   * Camera-following ocean grid: concentric rings with exponentially growing
   * radius. Dense (~1.4 m) at the center for 2 m skimming, reaching ~21 km at
   * the rim so the sea always meets the horizon.
   */
  private buildOceanGeometry(): THREE.BufferGeometry {
    const vertCount = 1 + OCEAN_RINGS * OCEAN_SEGMENTS
    const positions = new Float32Array(vertCount * 3)
    const wave = new Float32Array(vertCount).fill(1)
    const flow = new Float32Array(vertCount * 2)
    const edge = new Float32Array(vertCount)
    const depthA = new Float32Array(vertCount).fill(-1)
    // Slow uniform surface drift for the detail normals.
    for (let i = 0; i < vertCount; i++) {
      flow[i * 2] = 0.012
      flow[i * 2 + 1] = -0.045
    }

    let ptr = 3 // vertex 0 is the center at the origin
    for (let r = 0; r < OCEAN_RINGS; r++) {
      const radius = OCEAN_RING_BASE * Math.pow(OCEAN_RING_GROWTH, r)
      for (let s = 0; s < OCEAN_SEGMENTS; s++) {
        const a = (s / OCEAN_SEGMENTS) * Math.PI * 2
        positions[ptr++] = Math.cos(a) * radius
        positions[ptr++] = 0
        positions[ptr++] = Math.sin(a) * radius
      }
    }

    const idx = (r: number, s: number): number => 1 + r * OCEAN_SEGMENTS + (s % OCEAN_SEGMENTS)
    const indices = new Uint32Array(OCEAN_SEGMENTS * 3 + (OCEAN_RINGS - 1) * OCEAN_SEGMENTS * 6)
    let ip = 0
    for (let s = 0; s < OCEAN_SEGMENTS; s++) {
      indices[ip++] = 0
      indices[ip++] = idx(0, s + 1)
      indices[ip++] = idx(0, s)
    }
    for (let r = 0; r < OCEAN_RINGS - 1; r++) {
      for (let s = 0; s < OCEAN_SEGMENTS; s++) {
        const a = idx(r, s)
        const b = idx(r, s + 1)
        const c = idx(r + 1, s)
        const d = idx(r + 1, s + 1)
        indices[ip++] = a
        indices[ip++] = d
        indices[ip++] = c
        indices[ip++] = a
        indices[ip++] = b
        indices[ip++] = d
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aWaveScale', new THREE.BufferAttribute(wave, 1))
    geo.setAttribute('aFlow', new THREE.BufferAttribute(flow, 2))
    geo.setAttribute('aEdge', new THREE.BufferAttribute(edge, 1))
    geo.setAttribute('aDepth', new THREE.BufferAttribute(depthA, 1))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    return geo
  }

  /**
   * Trace the river channel down the valley (x ≈ 0, north to the sea) by
   * following the terrain minimum, then extrude a ribbon whose surface steps
   * monotonically downhill and whose edges tuck under the banks.
   */
  private buildRiverGeometry(ctx: GameContext): THREE.BufferGeometry | null {
    const stations: RiverStation[] = []
    let cx = 0
    let level = Number.POSITIVE_INFINITY
    let taper = -1 // becomes >= 0 once the river reaches sea level

    for (let z = -4250; z <= 3560; z += RIVER_STEP) {
      // Follow the channel: terrain minimum in a window around the last center.
      let bestH = Number.POSITIVE_INFINITY
      let bestX = cx
      for (let x = cx - 340; x <= cx + 340; x += 8) {
        const h = ctx.getTerrainHeight(x, z)
        if (h < bestH) {
          bestH = h
          bestX = x
        }
      }
      cx += (bestX - cx) * 0.25

      // Water never flows uphill: running minimum going downstream.
      level = Math.min(level, bestH + 1.7)
      if (taper < 0 && level <= 0.35) taper = 0
      const y = taper >= 0 ? Math.max(0.1, level) : level

      // Bank scan: widen until the terrain rises above the surface.
      const scanBank = (dir: 1 | -1): number => {
        let x = cx
        for (let i = 0; i < 40; i++) {
          x += dir * 4
          if (ctx.getTerrainHeight(x, z) > y + 0.45) break
        }
        return x + dir * 7
      }
      let xl = scanBank(-1)
      let xr = scanBank(1)
      if (xr - xl < 22) {
        const mid = (xl + xr) / 2
        xl = mid - 11
        xr = mid + 11
      }
      stations.push({ z, xl, xr, y })

      if (taper >= 0 && ++taper > 14) break // run ~220 m into the surf, then stop
    }
    if (stations.length < 2) return null

    const rows = stations.length
    const vertCount = rows * RIVER_ROW
    const positions = new Float32Array(vertCount * 3)
    const wave = new Float32Array(vertCount)
    const flow = new Float32Array(vertCount * 2)
    const edge = new Float32Array(vertCount)
    const depthA = new Float32Array(vertCount)

    for (let k = 0; k < rows; k++) {
      const st = stations[k]!
      // Downstream tangent from the centerline, for flow-aligned ripples.
      const prev = stations[Math.max(0, k - 1)]!
      const next = stations[Math.min(rows - 1, k + 1)]!
      const tx = (next.xl + next.xr - prev.xl - prev.xr) / 2
      const tz = next.z - prev.z
      const tl = Math.sqrt(tx * tx + tz * tz) || 1
      const fx = (tx / tl) * 0.3
      const fz = (tz / tl) * 0.3

      for (let j = 0; j < RIVER_ROW; j++) {
        const s = j / (RIVER_ROW - 1)
        const x = st.xl + (st.xr - st.xl) * s
        const th = ctx.getTerrainHeight(x, st.z)
        const isEdge = j === 0 || j === RIVER_ROW - 1
        const vi = k * RIVER_ROW + j
        positions[vi * 3] = x
        // Edge vertices dip below the bank so the ribbon never floats.
        positions[vi * 3 + 1] = isEdge ? Math.min(st.y, th - 1.1) : st.y
        positions[vi * 3 + 2] = st.z
        const e = Math.abs(s * 2 - 1)
        edge[vi] = e
        wave[vi] = RIVER_WAVE_SCALE * (1 - e * e)
        flow[vi * 2] = fx
        flow[vi * 2 + 1] = fz
        depthA[vi] = Math.max(st.y - th, 0)
      }
    }

    const indices = new Uint32Array((rows - 1) * (RIVER_ROW - 1) * 6)
    let ip = 0
    for (let k = 0; k < rows - 1; k++) {
      for (let j = 0; j < RIVER_ROW - 1; j++) {
        const a = k * RIVER_ROW + j
        const b = a + 1
        const c = a + RIVER_ROW
        const d = c + 1
        indices[ip++] = a
        indices[ip++] = d
        indices[ip++] = c
        indices[ip++] = a
        indices[ip++] = b
        indices[ip++] = d
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aWaveScale', new THREE.BufferAttribute(wave, 1))
    geo.setAttribute('aFlow', new THREE.BufferAttribute(flow, 2))
    geo.setAttribute('aEdge', new THREE.BufferAttribute(edge, 1))
    geo.setAttribute('aDepth', new THREE.BufferAttribute(depthA, 1))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    geo.computeBoundingSphere()
    return geo
  }
}
