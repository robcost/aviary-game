/**
 * Shared contract for every Aviary game module.
 *
 * World layout (all builder agents must honor this):
 * - The terrain is a square, {@link WORLD.size} meters on each edge, centered on the origin.
 * - Sea level is y = {@link WORLD.seaLevel}. The ocean lies to the south: terrain drops
 *   below sea level for z > ~3000.
 * - A mountain range rises in the north-west quadrant (around x ≈ -3500, z ≈ -3500),
 *   peaking near 900 m.
 * - A river valley runs roughly along x ≈ 0 from the ranges (z ≈ -4000) to the sea.
 * - Arid plateau country lies to the east (x > 2000).
 * - The player spawns at {@link WORLD.spawn}, flying south-ish over the valley.
 */
import type * as THREE from 'three'

/** Playable Australian species. */
export type SpeciesId =
  | 'wedge-tailed-eagle'
  | 'sulphur-crested-cockatoo'
  | 'rainbow-lorikeet'
  | 'galah'
  | 'laughing-kookaburra'

/** Display metadata for each species. Modules may extend this locally. */
export const SPECIES_INFO: Record<SpeciesId, { name: string }> = {
  'wedge-tailed-eagle': { name: 'Wedge-tailed Eagle' },
  'sulphur-crested-cockatoo': { name: 'Sulphur-crested Cockatoo' },
  'rainbow-lorikeet': { name: 'Rainbow Lorikeet' },
  galah: { name: 'Galah' },
  'laughing-kookaburra': { name: 'Laughing Kookaburra' },
}

/** World-scale constants shared by all modules. */
export const WORLD = {
  /** Terrain square edge length, meters. */
  size: 12000,
  /** Ocean plane height, meters. */
  seaLevel: 0,
  /** Player spawn position, meters. */
  spawn: { x: 0, y: 250, z: 1500 },
} as const

/** Live flight state. The Flight module owns it and writes it every frame. */
export interface FlightState {
  /** Bird position, world space, meters. */
  position: THREE.Vector3
  /** Bird velocity, world space, m/s. */
  velocity: THREE.Vector3
  /** Bird body orientation, world space. */
  quaternion: THREE.Quaternion
  /** Airspeed, m/s. */
  speed: number
  /** Height above the terrain surface, meters. */
  altitude: number
  /** Wing-beat cycle. Wraps in 0..1. Advances only while flapping. */
  flapPhase: number
  /** 0 = full glide, 1 = full-power flap. */
  flapStrength: number
  /** Bank amount: -1 (hard left) .. 1 (hard right). */
  banking: number
  /** True while the bird is below stall speed. */
  stalled: boolean
}

/** Shared context handed to every module. */
export interface GameContext {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  /** Seconds since boot. Does not advance while paused. */
  time: number
  /** Currently selected species. */
  species: SpeciesId
  /** True once the player leaves the menu (or in autofly/photo mode). */
  flying: boolean
  flight: FlightState
  /**
   * Normalized direction FROM the origin TOWARD the sun.
   * The Sky module owns and updates it; other modules read it.
   */
  sunDirection: THREE.Vector3
  /**
   * Exact terrain surface height at (x, z), meters.
   * The Terrain module replaces this at init. It must match the rendered
   * surface exactly — flight collision and object placement depend on it.
   */
  getTerrainHeight: (x: number, z: number) => number
  /**
   * Photo-mode preset index (1-based) from the `?shot=N` URL param, or null.
   * The Flight module applies the presets and freezes physics.
   */
  photoShot: number | null
  /** Scripted showcase flight (`?autofly=1`), used for review captures. */
  autofly: boolean
  paused: boolean
  /**
   * App-wide events.
   * - 'start-flight': CustomEvent<{ species: SpeciesId }> — fired by the UI menu.
   * - 'first-frame': Event — fired by main.ts exactly once, after the first
   *   full frame (all module updates plus the post-processing chain) has
   *   rendered. main.ts reveals the (until then hidden) canvas at this
   *   moment; the UI holds its boot backdrop until this fires, then plays
   *   its capped 1.5 s dissolve into the live scene.
   */
  events: EventTarget
}

/** A game subsystem. Instantiated once, init'd once, updated every frame. */
export interface GameModule {
  readonly name: string
  init(ctx: GameContext): void | Promise<void>
  update(dt: number, ctx: GameContext): void
}
