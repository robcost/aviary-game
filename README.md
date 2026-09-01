# Aviary

A first-person Australian bird flight simulator, built in [Three.js](https://threejs.org/).

Fly as one of five Australian birds — a Wedge-tailed Eagle, a Sulphur-crested
Cockatoo, a Rainbow Lorikeet, a Galah, or a Laughing Kookaburra — over a
12&nbsp;km procedural landscape: southern coastline, a river valley, a
north-west mountain range, and arid eastern plateau country, all under a
golden-hour sky. Every asset is procedural — no downloaded textures, models,
or sounds.

## Running locally

```bash
npm install
npm run dev
```

Then open the printed local URL in a browser.

## Controls

| Input | Action |
| --- | --- |
| Mouse | Steer (pitch / roll) |
| `Space` | Flap |
| `Shift` | Dive |
| `Esc` | Pause |

## Review / debug URL parameters

| Parameter | Effect |
| --- | --- |
| `?bird=<id>` | Preselect a species (e.g. `wedge-tailed-eagle`) |
| `?autofly=1` | Skip the menu and fly a scripted showcase path |
| `?shot=N` | Photo mode: freeze at preset camera `N` (1–4) |
| `?debug=1` | Show a performance overlay (draw calls, triangles, frame time) |

## Stack

Vite + strict TypeScript + [three](https://www.npmjs.com/package/three), with
[postprocessing](https://www.npmjs.com/package/postprocessing) and
[n8ao](https://www.npmjs.com/package/n8ao) for the render pipeline and
[simplex-noise](https://www.npmjs.com/package/simplex-noise) for terrain,
water, and vegetation generation.

## License

MIT — see [LICENSE](LICENSE).
