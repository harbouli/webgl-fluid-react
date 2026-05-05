# WebGL Fluid React

An interactive WebGL fluid simulation rendered inside a React app. Text is rasterized onto an off-screen canvas and used as a texture mask that the fluid dye follows, producing a liquid-text effect. Mouse/touch movement splats velocity and color into the simulation in real time.

## Features

- Real-time incompressible Navier-Stokes fluid simulation on the GPU
- Text rendered as a fluid-masked texture (font, size, and bold are configurable)
- Auto-preview animation plays until the user moves the pointer
- Live controls via [lil-gui](https://lil-gui.georgealways.com/) for font, color, and pointer size
- Built with React 19, TypeScript, and Vite

## How it works

The simulation runs entirely on the GPU through a series of WebGL fragment shader passes each frame:

1. **Splat** — injects velocity and dye color at the pointer position
2. **Divergence** — computes ∇·v of the velocity field
3. **Pressure** — solves the pressure Poisson equation (10 Jacobi iterations)
4. **Gradient subtract** — removes the pressure gradient to enforce ∇·v = 0
5. **Advection** — semi-Lagrangian self-advection of velocity and dye

The dye advection samples the text canvas texture so that color is pulled from the letter shapes.

## Getting started

```bash
# Install dependencies
bun install   # or npm install

# Start the dev server
bun dev       # or npm run dev

# Production build
bun run build # or npm run build
```

## Project structure

```
src/
  fluid.ts          # FluidSimulation class — WebGL setup, FBOs, per-frame step
  shaders.ts        # GLSL shader sources (vert + all frag shaders)
  FluidCanvas.tsx   # React component — mounts the canvas, wires up events & GUI
  App.tsx           # Root component
  main.tsx          # Entry point
```

## Configuration

The `FluidParams` object (exposed via the GUI) controls:

| Parameter     | Description                              |
|---------------|------------------------------------------|
| `text`        | String rendered as the fluid mask        |
| `fontName`    | One of the available font families       |
| `isBold`      | Bold weight toggle                       |
| `fontSize`    | Font size in logical pixels              |
| `pointerSize` | Radius of the velocity/color splat       |
| `color`       | RGB dye color injected at the pointer    |
