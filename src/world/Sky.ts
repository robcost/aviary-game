/**
 * Sky module — the full atmosphere and lighting rig for Aviary.
 *
 * Responsibilities:
 * - Physically-inspired scattering sky dome. Preetham-style analytic
 *   atmosphere (adapted from the three.js Sky example), hand-tuned for a low
 *   golden-hour sun over the 12 km Australian terrain.
 * - Visible HDR sun disc with a tight corona and a wide Mie forward-scatter
 *   halo, plus a sunward horizon glow band.
 * - A two-layer cloud system rendered inside the dome shader:
 *   - Cumulus: domain-warped FBM sampled on a world-anchored virtual plane
 *     (real parallax while flying), lit with a three-tap density gradient
 *     toward the sun for soft silver-lined rims, self-shadowed cores, and a
 *     powder-effect darkening. The shading is a two-light MIX keyed on sun
 *     access: the sun-facing side of every clump takes the deep-gold sun
 *     transmittance (the same per-pixel vFexSun the sky, fog tint, and
 *     horizon glow use), the shaded side falls to a cool blue-grey sky
 *     ambient — so clouds always participate in the sunset instead of
 *     reading as neutral grey smudges. A kilometre-scale macro field
 *     modulates coverage so the sky composes as broken banks and clear
 *     lanes rather than uniform noise, and a second slow field rotates the
 *     billow domain and varies the warp strength per bank, so separate
 *     banks carry genuinely different silhouettes (no repeated blob shape
 *     across views). The layer runs down to ~1 degree of elevation so
 *     perspective compresses distant banks against the horizon (the classic
 *     landscape scale cue), but its opacity thins near the horizon where
 *     the aerosol haze band already reads as low cloud; the
 *     aerial-perspective veil below takes them the rest of the way.
 *   - Cirrus: thin anisotropic streaks on a much higher plane.
 * - Aerial perspective: altitude-modulated {@link THREE.FogExp2} whose color
 *   tracks the horizon haze. The dome haze is additionally tinted by sun
 *   azimuth — warm on the sun side, cool blue-grey on the anti-sun side —
 *   so the horizon reads as lit air, not a flat gradient band. The tint
 *   relaxes toward the neutral fog color below the horizon so the terrain
 *   silhouette still dissolves seamlessly. Far ridges stack in warm layers and the
 *   terrain rim dissolves into the dome with no visible seam. Inside the dome
 *   shader the same haze is a Beer-Lambert veil through an exponential
 *   aerosol layer (optical depth ~ 1/sin(elevation), smooth-maxed through the
 *   horizon so the curve is C-infinity there, zenith depth subtracted). It
 *   decays over tens of degrees with no gradient-stop boundary and no
 *   derivative kink, and the final write is dithered with triangular-remapped
 *   interleaved gradient noise to kill 8-bit banding.
 * - The lighting rig: a warm shadow-casting sun {@link THREE.DirectionalLight}
 *   whose ortho shadow box follows the player camera, snapped to the
 *   shadow-map texel grid in light space (no edge shimmer), a sky/ground
 *   hemisphere fill, and `scene.environment` baked once from this very sky
 *   via PMREM so PBR materials receive true image-based lighting.
 *
 * This module OWNS `ctx.sunDirection`. It sets the direction at init and
 * re-asserts it every frame. All light placement, the shader sun, and the
 * shadow basis derive from the same vector.
 *
 * Per-frame cost: a handful of uniform writes and one snapped light
 * placement. Zero heap allocations after init — all scratch objects are
 * module-level.
 */
import * as THREE from 'three'
import type { GameContext, GameModule } from '../core/GameState'

/**
 * Frozen golden-hour sun direction (FROM the origin TOWARD the sun).
 * Low in the west-north-west (~12° elevation): it rims the north-west ranges
 * and rakes long shadows down the river valley for the spawn view.
 */
const SUN_DIRECTION = new THREE.Vector3(-0.8, 0.21, -0.5).normalize()

/** Shadow map resolution (one well-fed cascade). */
const SHADOW_MAP_SIZE = 4096
/** Half-extent of the shadow ortho box, meters, perpendicular to the light. */
const SHADOW_RADIUS = 500
/** Distance from the shadow anchor back along the sun direction, meters. */
const SUN_DISTANCE = 2600
/** How far ahead of the camera the shadow box is biased, meters. */
const SHADOW_LOOK_AHEAD = 260
/** Sky-dome radius, meters. Inside the camera far plane (40 km). */
const DOME_RADIUS = 30000

/** Fog color close to the ground: warm golden haze. */
const FOG_LOW = new THREE.Color(0xe6ba8a)
/** Fog color at altitude: cooler, lighter, thinner air. */
const FOG_HIGH = new THREE.Color(0xc3cfe0)

/**
 * Sky-dome vertex shader. Precomputes the scattering coefficients and the
 * sun-path transmittance parameters once per vertex.
 */
const SKY_VERTEX = /* glsl */ `
uniform vec3 sunPosition;
uniform float rayleigh;
uniform float turbidity;
uniform float mieCoefficient;
uniform vec3 up;

varying vec3 vWorldPosition;
varying vec3 vSunDirection;
varying float vSunfade;
varying vec3 vBetaR;
varying vec3 vBetaM;
varying float vSunE;
varying vec3 vFexSun;

const float e = 2.718281828459045;
const float pi = 3.141592653589793;

// Rayleigh scattering coefficients at sea level for (680, 550, 450) nm.
const vec3 totalRayleigh = vec3( 5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5 );

// Precomputed Mie term (K * w(lambda)).
const vec3 MieConst = vec3( 1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14 );

// Optical length scale heights, meters.
const float rayleighZenithLength = 8.4E3;
const float mieZenithLength = 1.25E3;

// Earth-shadow cutoff for the sun intensity curve.
const float cutoffAngle = 1.6110731556870734; // pi / 1.95
const float steepness = 1.5;
const float EE = 1000.0;

float sunIntensity( float zenithAngleCos ) {
  zenithAngleCos = clamp( zenithAngleCos, -1.0, 1.0 );
  return EE * max( 0.0, 1.0 - pow( e, -( ( cutoffAngle - acos( zenithAngleCos ) ) / steepness ) ) );
}

vec3 totalMie( float T ) {
  float c = ( 0.2 * T ) * 10E-18;
  return 0.434 * c * MieConst;
}

// Preetham relative optical length for a direction with cosine cosZenith.
float opticalScale( float cosZenith ) {
  float zenith = acos( max( 0.0, cosZenith ) );
  return 1.0 / ( cos( zenith ) + 0.15 * pow( 93.885 - ( zenith * 180.0 / pi ), -1.253 ) );
}

void main() {
  vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
  vWorldPosition = worldPosition.xyz;

  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  // Pin the dome to the far plane so it never depth-fights the world.
  gl_Position.z = gl_Position.w;

  vSunDirection = normalize( sunPosition );
  vSunE = sunIntensity( dot( vSunDirection, up ) );
  vSunfade = 1.0 - clamp( 1.0 - exp( sunPosition.y / 450000.0 ), 0.0, 1.0 );

  float rayleighCoefficient = rayleigh - ( 1.0 * ( 1.0 - vSunfade ) );
  vBetaR = totalRayleigh * rayleighCoefficient;
  vBetaM = totalMie( turbidity ) * mieCoefficient;

  // Transmittance along the SUN path (not the view path). This is the color
  // of direct sunlight after the atmosphere: deep gold at this elevation.
  // Used to light the clouds so they match the sky exactly.
  float sSun = opticalScale( dot( vSunDirection, up ) );
  vFexSun = exp( -( vBetaR * rayleighZenithLength * sSun + vBetaM * mieZenithLength * sSun ) );
}
`

/**
 * Sky-dome fragment shader. Per-pixel atmosphere, HDR sun disc with corona,
 * the two cloud layers, and the smooth air-mass horizon haze. Outputs linear
 * HDR; the renderer's ACES tone map and color-space conversion run through
 * the standard chunks at the end (and compile out during the PMREM bake),
 * followed by a triangular blue-noise-style dither against banding.
 */
const SKY_FRAGMENT = /* glsl */ `
uniform float mieDirectionalG;
uniform vec3 up;
uniform float uTime;
uniform float uCloudCoverage;
uniform float uCloudScale;
uniform float uCloudHeight;
uniform float uCirrusHeight;
uniform vec3 uHazeColor;

varying vec3 vWorldPosition;
varying vec3 vSunDirection;
varying float vSunfade;
varying vec3 vBetaR;
varying vec3 vBetaM;
varying float vSunE;
varying vec3 vFexSun;

const float pi = 3.141592653589793;

// Optical length scale heights, meters.
const float rayleighZenithLength = 8.4E3;
const float mieZenithLength = 1.25E3;

// cos of the sun radius. Slightly larger than reality for a fat, low sun.
const float sunAngularDiameterCos = 0.999936;

const float THREE_OVER_SIXTEENPI = 0.05968310365946075;
const float ONE_OVER_FOURPI = 0.07957747154594767;

float rayleighPhase( float cosTheta ) {
  return THREE_OVER_SIXTEENPI * ( 1.0 + cosTheta * cosTheta );
}

float hgPhase( float cosTheta, float g ) {
  float g2 = g * g;
  float inverse = 1.0 / pow( 1.0 - 2.0 * g * cosTheta + g2, 1.5 );
  return ONE_OVER_FOURPI * ( ( 1.0 - g2 ) * inverse );
}

// ---- 2D gradient noise + FBM for the cloud fields ---------------------------

vec2 hash2( vec2 p ) {
  p = vec2( dot( p, vec2( 127.1, 311.7 ) ), dot( p, vec2( 269.5, 183.3 ) ) );
  return -1.0 + 2.0 * fract( sin( p ) * 43758.5453123 );
}

float gnoise( in vec2 p ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  vec2 u = f * f * ( 3.0 - 2.0 * f );
  return mix(
    mix( dot( hash2( i + vec2( 0.0, 0.0 ) ), f - vec2( 0.0, 0.0 ) ),
         dot( hash2( i + vec2( 1.0, 0.0 ) ), f - vec2( 1.0, 0.0 ) ), u.x ),
    mix( dot( hash2( i + vec2( 0.0, 1.0 ) ), f - vec2( 0.0, 1.0 ) ),
         dot( hash2( i + vec2( 1.0, 1.0 ) ), f - vec2( 1.0, 1.0 ) ), u.x ),
    u.y );
}

// Full-detail FBM for the primary cloud shape.
float fbm5( vec2 p ) {
  float amp = 0.5;
  float sum = 0.0;
  for ( int i = 0; i < 5; i ++ ) {
    sum += amp * gnoise( p );
    p = p * 2.13 + vec2( 17.3, 9.1 );
    amp *= 0.5;
  }
  return sum;
}

// Cheaper FBM for the sun-occlusion taps and the cirrus layer.
float fbm3( vec2 p ) {
  float amp = 0.5;
  float sum = 0.0;
  for ( int i = 0; i < 3; i ++ ) {
    sum += amp * gnoise( p );
    p = p * 2.13 + vec2( 17.3, 9.1 );
    amp *= 0.5;
  }
  return sum;
}

// Coverage remap: FBM value -> soft-edged cloud density in 0..1.
// The warped fbm5 field measures roughly +-0.35, so the 1.35 gain expands it
// to ~[0.03, 0.97] before the threshold: clump cores genuinely reach density
// 1 (solid sunlit puffs) while the 0.30 width leaves wide soft skirts.
// (The old 0.62 gain compressed the field into [0.28, 0.72] against a full-
// density point of 0.73 — mean density 0.14, cores near-nonexistent — which
// rendered as an invisible haze, i.e. a cloudless sky.)
float shapeCloud( float f, float coverage ) {
  return smoothstep( coverage, coverage + 0.30, f * 1.35 + 0.5 );
}

void main() {
  vec3 direction = normalize( vWorldPosition - cameraPosition );

  // ---- atmosphere (Preetham) -----------------------------------------------

  float zenithAngle = acos( max( 0.0, dot( up, direction ) ) );
  float inverse = 1.0 / ( cos( zenithAngle ) + 0.15 * pow( 93.885 - ( ( zenithAngle * 180.0 ) / pi ), -1.253 ) );
  float sR = rayleighZenithLength * inverse;
  float sM = mieZenithLength * inverse;

  // Extinction along the view path.
  vec3 Fex = exp( -( vBetaR * sR + vBetaM * sM ) );

  // In-scattering.
  float cosTheta = dot( direction, vSunDirection );

  float rPhase = rayleighPhase( cosTheta * 0.5 + 0.5 );
  vec3 betaRTheta = vBetaR * rPhase;

  float mPhase = hgPhase( cosTheta, mieDirectionalG );
  vec3 betaMTheta = vBetaM * mPhase;

  vec3 Lin = pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * ( 1.0 - Fex ), vec3( 1.5 ) );
  Lin *= mix(
    vec3( 1.0 ),
    pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * Fex, vec3( 0.5 ) ),
    clamp( pow( 1.0 - dot( up, vSunDirection ), 5.0 ), 0.0, 1.0 ) );

  // Base radiance, HDR solar disc, and a tight corona for the bloom pass.
  vec3 L0 = vec3( 0.1 ) * Fex;
  float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00003, cosTheta );
  L0 += ( vSunE * 19000.0 * Fex ) * sundisk;
  float corona = pow( clamp( cosTheta, 0.0, 1.0 ), 600.0 );
  L0 += ( vSunE * 22.0 * Fex ) * corona;

  vec3 texColor = ( Lin + L0 ) * 0.04 + vec3( 0.0, 0.0003, 0.00075 );
  vec3 sky = pow( texColor, vec3( 1.0 / ( 1.2 + ( 1.2 * vSunfade ) ) ) );

  // Sunward horizon glow: a warm band that hugs the horizon near the sun,
  // selling the dust-laden golden-hour air over the plateau country.
  // Smooth |y|: abs() has a derivative kink at the horizon, and a C1 break
  // inside a smooth gradient reads as a Mach band.
  float absY = sqrt( direction.y * direction.y + 4.0e-4 );
  float horizonBand = pow( 1.0 - clamp( absY, 0.0, 1.0 ), 9.0 );
  float sunward = pow( clamp( cosTheta * 0.5 + 0.5, 0.0, 1.0 ), 5.0 );
  sky += vFexSun * vec3( 1.0, 0.55, 0.26 ) * ( horizonBand * sunward * 0.22 );

  // Direct-sun color for the cloud layers. Two corrections over the raw
  // ground-level transmittance vFexSun:
  // 1. Altitude: the cumulus deck at ~2.4 km (and cirrus at ~7 km) sees a far
  //    shorter aerosol/Rayleigh path than the ground, so its direct light is
  //    bright warm gold, not the deep blood-red of vFexSun. pow(Fex, k) with
  //    k < 1 is exactly exp(-k * tau): a principled shortening of the optical
  //    path, preserving the spectral shape.
  // 2. Level: the sky radiance above goes through the pow(1/2.4) perceptual
  //    lift, so surfaces shaded in raw-linear units render 2-3x darker than
  //    the sky they float in and disappear after tone mapping (the round-1
  //    "cloudless sky" bug in its second form). The 1.9 gain puts sunlit
  //    faces at ~1.3-1.6 HDR — clearly above the lifted sky — so tops read
  //    as bright gold-cream against blue, the classic cumulus statement.
  vec3 cloudSun  = pow( vFexSun, vec3( 0.55 ) ) * 1.9;
  vec3 cirrusSun = pow( vFexSun, vec3( 0.35 ) ) * 1.5;

  // ---- cumulus layer ---------------------------------------------------------

  // Clouds run down to ~1 degree of elevation: the perspective compression of
  // banks stacking against the horizon is the landscape scale cue the sky
  // needs. The clamped ray keeps the sample distance (and FBM frequency)
  // finite right at the horizon; the aerial haze below mutes the far field.
  float horizonFade = smoothstep( 0.008, 0.075, direction.y );
  if ( horizonFade > 0.001 ) {
    // Intersect the view ray with a virtual cloud plane above the camera.
    // The field is anchored in world XZ, so flying produces true parallax.
    float planeH = max( uCloudHeight - cameraPosition.y, 300.0 );
    float t = planeH / max( direction.y, 0.015 );
    vec2 p = ( cameraPosition.xz + direction.xz * t ) * uCloudScale;
    p += uTime * vec2( 0.0055, 0.002 ); // slow prevailing westerly

    // Kilometre-scale macro coverage: lowers the threshold into dense banks
    // in some regions and raises it into clear blue lanes in others, so the
    // cloudscape has composition instead of statistically uniform clumps.
    float macro = fbm3( p * 0.16 + vec2( 31.7, 7.7 ) );
    float coverage = uCloudCoverage - macro * 0.40;

    // Per-bank orientation and warp jitter: a slow kilometre-scale field
    // rotates the billow domain and varies the domain-warp strength across
    // the sky, so separate banks carry genuinely different silhouettes
    // instead of one repeated blob grammar (the "sprite variant with
    // rotation jitter" fix, done in the field itself).
    float bank = gnoise( p * 0.07 + vec2( 9.2, 3.1 ) );
    float ca = cos( bank * 2.6 );
    float sa = sin( bank * 2.6 );
    mat2 bankRot = mat2( ca, -sa, sa, ca );
    vec2 pr = bankRot * p;

    // Domain-warped FBM: q bends the field into billowing shapes.
    vec2 q = vec2( fbm5( pr ), fbm5( pr + vec2( 5.2, 1.3 ) ) );
    vec2 warped = pr + ( 1.35 + 0.80 * bank ) * q;
    float f = fbm5( warped );
    float density = shapeCloud( f, coverage );

    if ( density > 0.002 ) {
      // Two occlusion taps marching toward the sun in the cloud field. The
      // density gradient approximates how much cloud shades this sample:
      // thin sun-facing edges read as bright silver rims, thick cores darken.
      vec2 sunStep = normalize( vSunDirection.xz + vec2( 1.0e-4 ) );
      float d1 = shapeCloud( fbm3( warped + sunStep * 0.14 ), coverage );
      float d2 = shapeCloud( fbm3( warped + sunStep * 0.32 ), coverage );
      float occlusion = clamp( ( d1 * 0.65 + d2 * 0.35 ) - density * 0.15, 0.0, 1.0 );
      float sunAccess = 1.0 - occlusion * 0.85;

      // Powder effect: very thin cloud scatters less back toward the eye.
      float powder = 1.0 - exp( -density * 5.0 );

      // Two-stop lighting with real contrast: shaded bases sit clearly BELOW
      // the sky luminance (dark blue-grey silhouettes, anchored to the live
      // sky color so they read as objects inside this atmosphere), sunlit
      // faces sit clearly ABOVE it (bright altitude-gold). The mix is keyed
      // on the sun-occlusion gradient, squared to sharpen the terminator.
      vec3 cloudAmb = mix( vec3( 0.32, 0.38, 0.52 ), sky, 0.55 );
      vec3 shadedCol = cloudAmb * 0.58;
      vec3 litCol = cloudSun * ( 0.72 + 0.45 * powder );
      float lightMix = sunAccess * sunAccess;
      vec3 cloudCol = mix( shadedCol, litCol, lightMix );

      // Forward-scatter silver lining, strongest looking near the sun.
      float silver = pow( clamp( cosTheta, 0.0, 1.0 ), 8.0 );
      float rim = clamp( density - max( d1, d2 ), 0.0, 1.0 );
      cloudCol += cloudSun * silver * ( 1.8 * rim + 0.35 * sunAccess );

      // Thick occluded cores self-shadow toward a deep cool interior.
      cloudCol = mix( cloudCol, shadedCol * 0.82,
                      smoothstep( 0.55, 1.0, density ) * occlusion * 0.5 );

      // Gentle distance roll-off; the Beer-Lambert haze pass below does the
      // real work of sinking far banks into the horizon veil, so this only
      // needs to soften extreme-distance shimmer, not hide the clouds.
      // pow(density, 0.8) fattens the mid-density body so clumps read as
      // solid masses with soft skirts rather than translucent films.
      float distanceFade = exp( -t * 2.6e-5 );
      float alpha = pow( density, 0.8 ) * horizonFade * distanceFade;
      sky = mix( sky, cloudCol, alpha );
    }
  }

  // ---- cirrus layer ----------------------------------------------------------

  if ( horizonFade > 0.001 ) {
    float planeH = max( uCirrusHeight - cameraPosition.y, 1200.0 );
    float t = planeH / max( direction.y, 0.015 );
    vec2 p = ( cameraPosition.xz + direction.xz * t ) * ( uCloudScale * 0.45 );
    // Anisotropic sampling stretches the noise into wind-sheared streaks.
    p = vec2( p.x * 0.25 + p.y * 0.06, p.y * 1.35 );
    p += uTime * vec2( 0.0016, 0.0007 );
    // fbm3 spans roughly +-0.33; the 1.3 gain expands it so the smoothstep
    // carves distinct bright streaks out of the field instead of skimming the
    // barely-reachable top of a compressed range.
    float streak = fbm3( p ) * 1.3 + 0.5;
    float cirrus = smoothstep( 0.52, 0.95, streak ) * 0.6;
    float distanceFade = exp( -t * 1.6e-5 );
    vec3 cirrusCol = cirrusSun * ( 0.75 + 0.8 * pow( clamp( cosTheta, 0.0, 1.0 ), 4.0 ) )
                   + sky * 0.3;
    sky = mix( sky, cirrusCol, cirrus * horizonFade * distanceFade );
  }

  // ---- horizon haze ----------------------------------------------------------

  // Aerosol veil matched to the scene fog, modeled as an exponential aerosol
  // layer: optical depth along the view ray goes as 1 / sin(elevation).
  // The elevation runs through a smooth-max (C-infinity at the horizon —
  // a max()/abs() kink there reads as a Mach band), and the zenith optical
  // depth is subtracted so the veil vanishes exactly overhead. The resulting
  // Beer-Lambert weight decays over tens of degrees of elevation (0.98 at
  // the horizon, 0.82 at ~2 deg, 0.65 at ~5 deg, 0.45 at ~9 deg, 0.23 at
  // ~17 deg): no stop boundary and no steep shoulder anywhere for the eye
  // to catch as a stripe.
  float elevY = direction.y;
  float softElev = 0.5 * ( elevY + sqrt( elevY * elevY + 4.0e-4 ) );
  float tau = 0.46 * ( 1.0 / ( softElev * 3.6 + 0.10 ) - 1.0 / 3.7 );
  // Below the horizon the veil closes on the exact fog color so terrain and
  // ocean dissolve into the dome with no silhouette seam. The extra depth is
  // added smoothly in optical-depth space, not as a hard clamp on the weight.
  tau += 4.0 * ( 1.0 - smoothstep( -0.07, 0.02, elevY ) );
  float hazeW = 1.0 - exp( -tau );

  // Sun-azimuth hue gradient: the sun-side horizon warms toward amber, the
  // anti-sun side cools toward blue-grey (backscatter air). The tint relaxes
  // toward neutral below the horizon so the dome still converges on the exact
  // scene-fog color where terrain and ocean silhouettes dissolve into it.
  float dl = length( direction.xz ) + 1.0e-5;
  float cosAz = dot( direction.xz / dl, normalize( vSunDirection.xz ) );
  vec3 hazeTint = mix( vec3( 0.84, 0.91, 1.06 ), vec3( 1.10, 0.97, 0.86 ), cosAz * 0.5 + 0.5 );
  float tintAmt = mix( 0.30, 1.0, smoothstep( -0.05, 0.08, elevY ) );
  vec3 hazeCol = uHazeColor * ( 0.92 + 0.35 * sunward ) * mix( vec3( 1.0 ), hazeTint, tintAmt );
  // The thin end of the veil takes its hue from the scattering solution, so
  // high up the haze reads as the same air as the sky rather than a painted
  // overlay; only the saturated horizon end converges on the flat fog color.
  hazeCol = mix( sky, hazeCol, 0.55 + 0.45 * hazeW );
  sky = mix( sky, hazeCol, hazeW );

  gl_FragColor = vec4( sky, 1.0 );

  #include <tonemapping_fragment>
  #include <colorspace_fragment>

  // Triangular-remapped interleaved gradient noise, applied after tone map +
  // color-space conversion: breaks up 8-bit quantization banding in the large
  // smooth sky gradients with sub-LSB noise (invisible as grain).
  float ign = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) );
  float dither = ign * 2.0 - 1.0;
  dither = sign( dither ) * ( 1.0 - sqrt( 1.0 - abs( dither ) ) );
  gl_FragColor.rgb += dither * ( 1.0 / 255.0 );
}
`

// Module-level scratch. Reused every frame; never allocated in update().
const _forward = new THREE.Vector3()
const _anchor = new THREE.Vector3()
const _origin = new THREE.Vector3()
const _up = new THREE.Vector3(0, 1, 0)

/**
 * Atmosphere, clouds, fog, and the complete lighting rig.
 * See the module doc comment at the top of this file for the full breakdown.
 */
export class Sky implements GameModule {
  readonly name = 'sky'

  /** Frozen sun direction — the single source of truth for the whole rig. */
  private readonly sunDir = new THREE.Vector3()

  /** Shader uniforms for the sky dome. */
  private readonly uniforms = {
    sunPosition: { value: new THREE.Vector3() },
    up: { value: new THREE.Vector3(0, 1, 0) },
    rayleigh: { value: 2.8 },
    turbidity: { value: 8.0 },
    mieCoefficient: { value: 0.0055 },
    mieDirectionalG: { value: 0.82 },
    uTime: { value: 0 },
    // Base coverage threshold (lower = more cloud). The macro FBM in the
    // shader swings it +-~0.13 across kilometres for banks and clear lanes.
    // Tuned visually against live photo-mode captures: a broken-cumulus sky
    // with solid gold-topped banks, dark bases, and clear blue lanes —
    // not overcast, not clear.
    uCloudCoverage: { value: 0.41 },
    uCloudScale: { value: 0.00052 },
    uCloudHeight: { value: 2400 },
    uCirrusHeight: { value: 7200 },
    uHazeColor: { value: new THREE.Color() },
  }

  private dome!: THREE.Mesh<THREE.BoxGeometry, THREE.ShaderMaterial>
  private sun!: THREE.DirectionalLight
  private fog!: THREE.FogExp2

  /** Rotation from light space to world space (z axis = sun direction). */
  private readonly lightBasis = new THREE.Matrix4()
  /** Rotation from world space to light space. */
  private readonly lightBasisInv = new THREE.Matrix4()

  /**
   * Build the dome, bake the PMREM environment, and install the light rig.
   * Sets `ctx.sunDirection` for every module that reads it afterward.
   */
  init(ctx: GameContext): void {
    this.sunDir.copy(SUN_DIRECTION)
    ctx.sunDirection.copy(this.sunDir)
    this.uniforms.sunPosition.value.copy(this.sunDir)

    // -- sky dome --------------------------------------------------------------
    const material = new THREE.ShaderMaterial({
      name: 'AviarySky',
      uniforms: this.uniforms,
      vertexShader: SKY_VERTEX,
      fragmentShader: SKY_FRAGMENT,
      side: THREE.BackSide,
      depthWrite: false,
    })
    this.dome = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material)
    this.dome.scale.setScalar(DOME_RADIUS)
    this.dome.frustumCulled = false
    // Draw the sky after the opaque world: early-z rejects every occluded
    // pixel, so the (comparatively expensive) cloud FBM only runs on sky.
    this.dome.renderOrder = 1000

    // -- aerial perspective ------------------------------------------------------
    this.fog = new THREE.FogExp2(FOG_LOW.getHex(), 0.00014)
    ctx.scene.fog = this.fog
    ctx.scene.background = new THREE.Color(FOG_LOW.getHex())
    this.updateFog(ctx)

    // -- environment: bake THIS sky through PMREM so PBR gets real IBL ----------
    // PMREMGenerator disables tone mapping during the bake, so the dome's
    // tonemapping chunk compiles out and the environment stays linear HDR.
    const pmrem = new THREE.PMREMGenerator(ctx.renderer)
    const bakeScene = new THREE.Scene()
    bakeScene.add(this.dome)
    const envRT = pmrem.fromScene(bakeScene, 0.035, 1, DOME_RADIUS * 2)
    ctx.scene.environment = envRT.texture
    ctx.scene.environmentIntensity = 0.55
    pmrem.dispose()

    ctx.scene.add(this.dome) // re-parents out of the bake scene

    // -- lighting rig ------------------------------------------------------------
    const hemi = new THREE.HemisphereLight(0x84a2c8, 0x9a7044, 0.38)

    this.sun = new THREE.DirectionalLight(0xffd29b, 3.2)
    this.sun.castShadow = true
    this.sun.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
    const shadowCam = this.sun.shadow.camera
    shadowCam.left = -SHADOW_RADIUS
    shadowCam.right = SHADOW_RADIUS
    shadowCam.top = SHADOW_RADIUS
    shadowCam.bottom = -SHADOW_RADIUS
    shadowCam.near = 1
    shadowCam.far = SUN_DISTANCE * 2
    shadowCam.updateProjectionMatrix()
    this.sun.shadow.bias = -0.0001
    this.sun.shadow.normalBias = 2.0
    ctx.scene.add(hemi, this.sun, this.sun.target)

    // Fixed light basis for texel snapping. Same lookAt convention (and up
    // vector) as the shadow camera, so the quantization grid aligns with the
    // shadow map's texel grid exactly.
    this.lightBasis.lookAt(this.sunDir, _origin, _up)
    this.lightBasisInv.copy(this.lightBasis).invert()
  }

  /**
   * Per frame: ride the dome on the camera, drift the clouds, modulate the
   * fog with altitude, and re-place the snapped shadow box. Zero allocations.
   */
  update(_dt: number, ctx: GameContext): void {
    // This module owns the sun direction; hold it steady against any drift.
    ctx.sunDirection.copy(this.sunDir)

    // The dome rides on the camera so its shell never comes into reach.
    this.dome.position.copy(ctx.camera.position)
    this.uniforms.uTime.value = ctx.time

    this.updateFog(ctx)
    this.updateShadowRig(ctx)
  }

  /**
   * Altitude-aware fog: dense warm haze down in the valleys, thinner and
   * cooler air up high. The dome's below-horizon color follows the same
   * value, so terrain always dissolves cleanly into the sky.
   */
  private updateFog(ctx: GameContext): void {
    const h = THREE.MathUtils.clamp(ctx.camera.position.y, 0, 4000)
    this.fog.density = 0.000125 * Math.exp(-h / 1800) + 0.00003
    this.fog.color.lerpColors(FOG_LOW, FOG_HIGH, THREE.MathUtils.clamp(h / 3000, 0, 1))
    this.uniforms.uHazeColor.value.copy(this.fog.color)
  }

  /**
   * Keep the shadow ortho box centered ahead of the camera, snapped to the
   * shadow-map texel grid in light space. The light's rotation never changes,
   * so snapping removes all shadow-edge shimmer while flying.
   */
  private updateShadowRig(ctx: GameContext): void {
    ctx.camera.getWorldDirection(_forward)
    _anchor.copy(ctx.camera.position).addScaledVector(_forward, SHADOW_LOOK_AHEAD)

    // World -> light space, quantize the two axes across the light, back out.
    _anchor.applyMatrix4(this.lightBasisInv)
    const texel = (SHADOW_RADIUS * 2) / SHADOW_MAP_SIZE
    _anchor.x = Math.floor(_anchor.x / texel) * texel
    _anchor.y = Math.floor(_anchor.y / texel) * texel
    _anchor.applyMatrix4(this.lightBasis)

    this.sun.target.position.copy(_anchor)
    this.sun.position.copy(_anchor).addScaledVector(this.sunDir, SUN_DISTANCE)
  }
}
