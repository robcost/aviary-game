/**
 * Aviary presentation layer: cinematic title screen, glassmorphic in-flight
 * HUD, pause overlay, launch transition, and the boot fade.
 *
 * Responsibilities:
 * - Title screen with species-select cards (name, Latin name, character
 *   blurb, stat bars, per-species accent colour). Fires the `'start-flight'`
 *   CustomEvent on {@link GameContext.events} when the player picks a bird.
 * - A warm "launch bloom" veil that carries the eye from menu to flight.
 * - In-flight HUD: airspeed, altitude, compass ribbon, heading readout,
 *   species name, and a stall warning. The HUD fades out after a few seconds
 *   of input idleness and returns on any input (or on a stall).
 * - Pause overlay, toggled with Escape while the pointer is not locked.
 *   This module owns {@link GameContext.paused}.
 * - A 1.5 s fade-from-black on boot.
 *
 * The menu never renders in autofly/photo modes (`ctx.flying` is already
 * true at init), and the HUD stays hidden in photo mode so review captures
 * stay clean. All visual styling lives in `index.html`; this module only
 * builds DOM and drives per-frame values with zero heap allocations on the
 * steady-state path (numeric strings are written only when a displayed
 * value actually changes).
 */
import * as THREE from 'three'
import { SPECIES_INFO, type GameContext, type GameModule, type SpeciesId } from '../core/GameState'

/** Card metadata for one selectable species. Stats are 0–100. */
interface SpeciesCard {
  /** Scientific (Latin) name, shown under the common name. */
  latin: string
  /** One-line character blurb. */
  blurb: string
  /** Top-end level flight speed, relative. */
  speed: number
  /** Roll/turn responsiveness, relative. */
  agility: number
  /** Soaring efficiency — how long it holds altitude without flapping. */
  glide: number
  /** Accent colour for the card's glyph, hover ring, and stat bars. */
  accent: string
  /** Soft rgba() version of the accent, used for glows. */
  glow: string
}

/**
 * Honest relative stats: the eagle is a fast soarer, the lorikeet a darting
 * sprinter, the kookaburra a short-hop perch hunter. Accents echo each
 * bird's plumage.
 */
const SPECIES_CARDS: Record<SpeciesId, SpeciesCard> = {
  'wedge-tailed-eagle': {
    latin: 'Aquila audax',
    blurb: "Australia's largest raptor. A two-metre span that owns the thermals.",
    speed: 84,
    agility: 46,
    glide: 96,
    accent: '#e8c47a',
    glow: 'rgba(232, 196, 122, 0.45)',
  },
  'sulphur-crested-cockatoo': {
    latin: 'Cacatua galerita',
    blurb: 'Loud, brilliant white, and smarter than it lets on.',
    speed: 62,
    agility: 70,
    glide: 52,
    accent: '#ffe9a8',
    glow: 'rgba(255, 233, 168, 0.42)',
  },
  'rainbow-lorikeet': {
    latin: 'Trichoglossus moluccanus',
    blurb: 'A darting streak of colour with a nectar habit and no patience.',
    speed: 78,
    agility: 95,
    glide: 34,
    accent: '#63cdaa',
    glow: 'rgba(99, 205, 170, 0.42)',
  },
  galah: {
    latin: 'Eolophus roseicapilla',
    blurb: 'Pink-and-grey larrikin of the open country. Flies for the fun of it.',
    speed: 70,
    agility: 80,
    glide: 46,
    accent: '#f2a4b8',
    glow: 'rgba(242, 164, 184, 0.42)',
  },
  'laughing-kookaburra': {
    latin: 'Dacelo novaeguineae',
    blurb: 'The bush’s laughing sentry. A patient hunter of short, deadly hops.',
    speed: 52,
    agility: 62,
    glide: 38,
    accent: '#93bede',
    glow: 'rgba(147, 190, 222, 0.42)',
  },
}

/**
 * Stylized wing mark: two swept strokes reading as a bird on the wing.
 * Tinted via `currentColor` so each card can colour it with its accent.
 */
const WING_MARK =
  '<svg class="wing-mark" viewBox="0 0 64 32" fill="none" stroke="currentColor" aria-hidden="true">' +
  '<path d="M6 22 Q19 7 32 18.5 Q45 7 58 22" stroke-width="2.2" stroke-linecap="round"/>' +
  '<path d="M16 27 Q25 19.5 32 24.5 Q39 19.5 48 27" stroke-width="1.3" stroke-linecap="round" opacity="0.45"/>' +
  '</svg>'

/** Compass ribbon: pixels per degree of heading. */
const PX_PER_DEG = 3
/** Seconds of input idleness before the HUD fades out. */
const HUD_IDLE_SECONDS = 4
/** Cardinal/intercardinal labels, indexed by degrees/45. */
const CARDINALS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const

/**
 * The UI game module. Instantiated once by `main.ts`; builds all DOM in
 * {@link UI.init} and drives dynamic HUD values in {@link UI.update}.
 */
export class UI implements GameModule {
  readonly name = 'ui'

  private menu: HTMLDivElement | null = null
  private hud!: HTMLDivElement
  private speedValue!: HTMLSpanElement
  private altValue!: HTMLSpanElement
  private speciesLabel!: HTMLDivElement
  private ribbon!: HTMLDivElement
  private headingLabel!: HTMLDivElement
  private stallLabel!: HTMLDivElement
  private pause!: HTMLDivElement
  private pauseSpecies!: HTMLDivElement

  /** Seconds since the last player input; drives the HUD idle fade. */
  private idleSeconds = 0
  /** Half the compass viewport width, px; cached on resize. */
  private compassHalfWidth = 0

  // Last-written display values, used to skip redundant DOM writes.
  private lastSpeed = -1
  private lastAlt = -1
  private lastHeading = -1
  private lastSpecies: SpeciesId | null = null
  private lastStalled = false
  private hudOn = false
  private hudIdle = false

  /** Scratch vector for heading math. Reused every frame; never reallocated. */
  private readonly scratchForward = new THREE.Vector3()

  /** Build all UI DOM and wire input listeners. */
  init(ctx: GameContext): void {
    const root = document.getElementById('ui')!

    this.buildHud(root, ctx)
    this.buildPause(root, ctx)
    if (!ctx.flying) this.buildMenu(root, ctx)
    if (ctx.photoShot === null) this.playBootFade(root, ctx)

    // Any input wakes the HUD from its idle fade.
    const wake = (): void => {
      this.idleSeconds = 0
    }
    addEventListener('pointermove', wake, { passive: true })
    addEventListener('pointerdown', wake, { passive: true })
    addEventListener('wheel', wake, { passive: true })
    addEventListener('keydown', (e) => {
      wake()
      if (
        e.code === 'Escape' &&
        ctx.flying &&
        ctx.photoShot === null &&
        document.pointerLockElement === null
      ) {
        this.setPaused(ctx, !ctx.paused)
      }
    })

    const measure = (): void => {
      this.compassHalfWidth = this.ribbon.parentElement!.clientWidth / 2
    }
    addEventListener('resize', measure)
    measure()
  }

  /** Per-frame HUD refresh. Steady-state path allocates nothing. */
  update(dt: number, ctx: GameContext): void {
    const showHud = ctx.flying && ctx.photoShot === null
    if (showHud !== this.hudOn) {
      this.hudOn = showHud
      this.hud.classList.toggle('on', showHud)
    }
    if (!showHud) return

    const f = ctx.flight

    // A stall always wakes the HUD — the warning must never fade away.
    if (f.stalled) this.idleSeconds = 0

    // Idle fade: hide after a few seconds without input (not while paused).
    if (!ctx.paused) this.idleSeconds += dt
    const idle = this.idleSeconds > HUD_IDLE_SECONDS || ctx.paused
    if (idle !== this.hudIdle) {
      this.hudIdle = idle
      this.hud.classList.toggle('idle', idle)
    }

    const speed = Math.round(f.speed * 3.6)
    if (speed !== this.lastSpeed) {
      this.lastSpeed = speed
      this.speedValue.textContent = String(speed)
    }

    const alt = Math.max(0, Math.round(f.altitude))
    if (alt !== this.lastAlt) {
      this.lastAlt = alt
      this.altValue.textContent = String(alt)
    }

    // Heading: 0° = north (-Z), 90° = east (+X).
    const fwd = this.scratchForward.set(0, 0, -1).applyQuaternion(f.quaternion)
    let heading = (Math.atan2(fwd.x, -fwd.z) * 180) / Math.PI
    if (heading < 0) heading += 360
    const headingTenths = Math.round(heading * 10)
    if (headingTenths !== this.lastHeading) {
      this.lastHeading = headingTenths
      const x = this.compassHalfWidth - (heading + 180) * PX_PER_DEG
      this.ribbon.style.transform = `translate3d(${x.toFixed(1)}px,0,0)`
      this.headingLabel.textContent = `${String(Math.round(heading) % 360).padStart(3, '0')}°`
    }

    if (ctx.species !== this.lastSpecies) {
      this.lastSpecies = ctx.species
      this.speciesLabel.textContent = SPECIES_INFO[ctx.species].name
      this.pauseSpecies.textContent = SPECIES_INFO[ctx.species].name
    }

    if (f.stalled !== this.lastStalled) {
      this.lastStalled = f.stalled
      this.stallLabel.classList.toggle('on', f.stalled)
    }
  }

  /**
   * Create the boot fade-from-black overlay and retire it when done.
   *
   * The overlay stays fully opaque until main.ts fires 'first-frame' (the
   * first complete render of the whole pipeline). Only then does the capped
   * 1.5 s fade start, so the reveal always lands on a finished frame — never
   * on a black, still-compiling canvas.
   */
  private playBootFade(root: HTMLElement, ctx: GameContext): void {
    const fade = document.createElement('div')
    fade.className = 'boot-fade'
    root.appendChild(fade)
    ctx.events.addEventListener(
      'first-frame',
      () => {
        // Two frames so the presented render and the committed opacity:1
        // both precede the transition start.
        requestAnimationFrame(() => {
          requestAnimationFrame(() => fade.classList.add('gone'))
        })
      },
      { once: true },
    )
    fade.addEventListener('transitionend', () => fade.remove())
  }

  /** Build the in-flight HUD (hidden until `ctx.flying`). */
  private buildHud(root: HTMLElement, ctx: GameContext): void {
    this.hud = document.createElement('div')
    this.hud.className = 'hud'
    this.hud.innerHTML =
      // The edge-fade mask lives on .compass-window, NOT on .compass itself:
      // a mask on the glass element would clip its backdrop-filter scrim and
      // wash the strip out against a blown sky.
      '<div class="compass hud-glass">' +
      '<div class="compass-window">' +
      '<div class="compass-ribbon"></div>' +
      '</div>' +
      '<div class="compass-caret"></div>' +
      '</div>' +
      '<div class="compass-heading">000°</div>' +
      '<div class="hud-metric hud-speed hud-glass">' +
      '<span class="value">0</span><span class="label">km/h</span></div>' +
      '<div class="hud-metric hud-alt hud-glass">' +
      '<span class="value">0</span><span class="label">metres</span></div>' +
      '<div class="hud-species hud-glass"></div>' +
      '<div class="hud-stall">Stall</div>'
    root.appendChild(this.hud)

    this.ribbon = this.hud.querySelector<HTMLDivElement>('.compass-ribbon')!
    this.headingLabel = this.hud.querySelector<HTMLDivElement>('.compass-heading')!
    this.speedValue = this.hud.querySelector<HTMLSpanElement>('.hud-speed .value')!
    this.altValue = this.hud.querySelector<HTMLSpanElement>('.hud-alt .value')!
    this.speciesLabel = this.hud.querySelector<HTMLDivElement>('.hud-species')!
    this.stallLabel = this.hud.querySelector<HTMLDivElement>('.hud-stall')!
    this.speciesLabel.textContent = SPECIES_INFO[ctx.species].name

    this.buildCompassTicks()
  }

  /**
   * Populate the compass ribbon with ticks and labels covering −180°..540°,
   * so any translation for a 0–360° heading shows a seamless strip.
   */
  private buildCompassTicks(): void {
    const frag = document.createDocumentFragment()
    for (let deg = -180; deg <= 540; deg += 5) {
      const norm = ((deg % 360) + 360) % 360
      const tick = document.createElement('div')
      const major = norm % 45 === 0
      tick.className = major ? 'compass-tick major' : 'compass-tick'
      tick.style.left = `${(deg + 180) * PX_PER_DEG}px`
      frag.appendChild(tick)
      if (major) {
        const label = document.createElement('div')
        const isCardinal = norm % 90 === 0
        label.className = isCardinal ? 'compass-label cardinal' : 'compass-label'
        label.textContent = CARDINALS[norm / 45]
        label.style.left = `${(deg + 180) * PX_PER_DEG}px`
        frag.appendChild(label)
      }
    }
    this.ribbon.appendChild(frag)
  }

  /** Build the pause overlay (hidden until toggled). */
  private buildPause(root: HTMLElement, ctx: GameContext): void {
    this.pause = document.createElement('div')
    this.pause.className = 'pause panel'
    this.pause.innerHTML =
      WING_MARK +
      '<h2 class="pause-title">Paused</h2>' +
      '<div class="pause-species"></div>' +
      '<div class="pause-rule"></div>' +
      '<div class="pause-hints">' +
      '<span><kbd>Mouse</kbd>Steer</span>' +
      '<span><kbd>Space</kbd>Flap</span>' +
      '<span><kbd>Shift</kbd>Dive</span>' +
      '</div>' +
      '<div class="pause-actions">' +
      '<button class="pause-resume" type="button">Resume</button>' +
      '<button class="pause-change-bird" type="button">Change Bird</button>' +
      '</div>' +
      '<div class="pause-hint">Esc to resume</div>'
    root.appendChild(this.pause)

    this.pauseSpecies = this.pause.querySelector<HTMLDivElement>('.pause-species')!
    this.pauseSpecies.textContent = SPECIES_INFO[ctx.species].name
    this.pause
      .querySelector<HTMLButtonElement>('.pause-resume')!
      .addEventListener('click', () => this.setPaused(ctx, false))
    // A full reload is the simplest correct way back to the menu: every
    // module owns one-time init state keyed to the species/world, and a
    // fresh load re-runs it cleanly with no query params (so autofly/photo
    // modes don't re-trigger and the title screen shows).
    this.pause
      .querySelector<HTMLButtonElement>('.pause-change-bird')!
      .addEventListener('click', () => {
        location.href = location.pathname
      })
  }

  /** Toggle the pause state and overlay. This module owns `ctx.paused`. */
  private setPaused(ctx: GameContext, paused: boolean): void {
    ctx.paused = paused
    this.pause.classList.toggle('on', paused)
  }

  /** Build the title screen with the species-select cards. */
  private buildMenu(root: HTMLElement, ctx: GameContext): void {
    const menu = document.createElement('div')
    menu.className = 'menu panel'
    this.menu = menu

    const head = document.createElement('header')
    head.className = 'menu-head'
    head.innerHTML =
      `<div class="menu-mark">${WING_MARK}</div>` +
      '<div class="menu-eyebrow">A flight over country</div>' +
      '<h1 class="menu-title">AVIARY</h1>' +
      '<div class="menu-rule"></div>' +
      '<div class="menu-sub">Choose your bird</div>'
    menu.appendChild(head)

    const cards = document.createElement('div')
    cards.className = 'cards'
    const ids = Object.keys(SPECIES_CARDS) as SpeciesId[]
    const buttons: HTMLButtonElement[] = []
    ids.forEach((id, i) => {
      const btn = this.buildCard(id, i, () => this.launch(ctx, id, btn))
      buttons.push(btn)
      cards.appendChild(btn)
    })
    menu.appendChild(cards)

    const hints = document.createElement('footer')
    hints.className = 'menu-hints'
    hints.innerHTML =
      '<span><kbd>Mouse</kbd>Steer</span>' +
      '<span><kbd>Space</kbd>Flap</span>' +
      '<span><kbd>Shift</kbd>Dive</span>' +
      '<span><kbd>Esc</kbd>Pause</span>'
    menu.appendChild(hints)

    root.appendChild(menu)

    // Focus the pre-selected species (?bird=) and enable arrow-key browsing.
    const preselect = Math.max(0, ids.indexOf(ctx.species))
    buttons[preselect].focus()
    menu.addEventListener('keydown', (e) => {
      if (e.code !== 'ArrowLeft' && e.code !== 'ArrowRight') return
      const active = buttons.indexOf(document.activeElement as HTMLButtonElement)
      const from = active >= 0 ? active : preselect
      const step = e.code === 'ArrowRight' ? 1 : -1
      buttons[(from + step + buttons.length) % buttons.length].focus()
      e.preventDefault()
    })

    // Stagger the cards in after the DOM commits the initial state.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => cards.classList.add('in'))
    })
  }

  /** Build one species card button. */
  private buildCard(id: SpeciesId, index: number, onSelect: () => void): HTMLButtonElement {
    const meta = SPECIES_CARDS[id]
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'card'
    btn.style.setProperty('--i', String(index))
    btn.style.setProperty('--accent', meta.accent)
    btn.style.setProperty('--accent-glow', meta.glow)
    btn.innerHTML =
      `<span class="card-glyph">${WING_MARK}</span>` +
      `<span class="card-name">${SPECIES_INFO[id].name}` +
      `<span class="card-latin">${meta.latin}</span></span>` +
      `<span class="card-blurb">${meta.blurb}</span>` +
      '<span class="card-stats">' +
      this.buildStat('Speed', meta.speed) +
      this.buildStat('Agility', meta.agility) +
      this.buildStat('Glide', meta.glide) +
      '</span>'
    btn.addEventListener('click', onSelect)
    return btn
  }

  /** Markup for one labelled stat bar. */
  private buildStat(label: string, value: number): string {
    return (
      `<span class="stat"><span class="stat-label">${label}</span>` +
      `<span class="stat-track"><span class="stat-fill" style="--v:${value}%"></span></span></span>`
    )
  }

  /**
   * Fade the menu into flight: glow the chosen card, sweep a warm launch
   * bloom across the screen, and fire the `'start-flight'` event under it.
   */
  private launch(ctx: GameContext, species: SpeciesId, card: HTMLButtonElement): void {
    const menu = this.menu
    if (!menu || menu.classList.contains('exit')) return
    card.classList.add('selected')
    menu.classList.add('exit')
    menu.style.pointerEvents = 'none'

    // Warm bloom veil: rises as the menu leaves, then dissolves into flight.
    const veil = document.createElement('div')
    veil.className = 'launch-veil'
    menu.parentElement!.appendChild(veil)
    veil.addEventListener('animationend', () => veil.remove())

    // Let the selected-card glow land before the world takes over.
    setTimeout(() => {
      ctx.events.dispatchEvent(new CustomEvent('start-flight', { detail: { species } }))
      this.idleSeconds = 0
    }, 350)
    setTimeout(() => {
      menu.remove()
      this.menu = null
    }, 1400)
  }
}
