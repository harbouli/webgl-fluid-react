import {
  vertShader,
  fragShaderAdvection,
  fragShaderDivergence,
  fragShaderPressure,
  fragShaderGradientSubtract,
  fragShaderPoint,
  fragShaderOutputShader,
} from "./shaders";

export interface FluidParams {
  fontName: string;
  isBold: boolean;
  fontSize: number;
  text: string;
  pointerSize: number;
  color: { r: number; g: number; b: number };
}

export const fontOptions: Record<string, string> = {
  Arial: "Arial, sans-serif",
  Verdana: "Verdana, sans-serif",
  Tahoma: "Tahoma, sans-serif",
  "Times New Roman": "Times New Roman, serif",
  Georgia: "Georgia, serif",
  Garamond: "Garamond, serif",
  "Courier New": "Courier New, monospace",
  "Brush Script MT": "Brush Script MT, cursive",
};

interface FBO {
  fbo: WebGLFramebuffer;
  width: number;
  height: number;
  attach(id: number): number;
}

interface DoubleFBO {
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  read(): FBO;
  write(): FBO;
  swap(): void;
}

interface ShaderProgram {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation>;
}

export class FluidSimulation {
  private canvas: HTMLCanvasElement; // WebGL render target
  private gl: WebGLRenderingContext;
  // Off-screen canvas used to rasterize text into a WebGL texture
  private textureCanvas: HTMLCanvasElement;
  private textureCtx: CanvasRenderingContext2D;
  // GPU texture holding the current text raster, sampled by the advection shader
  private canvasTexture!: WebGLTexture;

  // Injects velocity/color at the pointer position (Gaussian splat)
  private splatProgram!: ShaderProgram;
  // Computes the divergence of the velocity field (∇·v)
  private divergenceProgram!: ShaderProgram;
  // Solves the pressure Poisson equation iteratively (Jacobi iterations)
  private pressureProgram!: ShaderProgram;
  // Subtracts the pressure gradient to enforce incompressibility (∇·v = 0)
  private gradientSubtractProgram!: ShaderProgram;
  // Semi-Lagrangian advection — moves quantities along the velocity field
  private advectionProgram!: ShaderProgram;
  // Composites the fluid color buffer onto the screen
  private outputShaderProgram!: ShaderProgram;

  // Ping-pong buffer for the dye/color field
  private outputColor!: DoubleFBO;
  // Ping-pong buffer for the 2-D velocity field (stored as RG channels)
  private velocity!: DoubleFBO;
  // Single-write buffer for the velocity divergence (read-only after each step)
  private divergence!: FBO;
  // Ping-pong buffer for the pressure field used in the projection step
  private pressure!: DoubleFBO;

  // Current pointer state; dx/dy are frame-to-frame deltas used as splat force
  private pointer = { x: 0, y: 0, dx: 0, dy: 0, moved: false };
  // True until the user first interacts; drives the auto-preview animation
  private isPreview = true;
  // requestAnimationFrame handle, kept so stop() can cancel the loop
  private rafId = 0;

  params: FluidParams = {
    fontName: "Verdana",
    isBold: false,
    fontSize: 80,
    text: "Mohamed",
    pointerSize: 0,
    color: { r: 1, g: 0, b: 0.5 },
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;
    gl.getExtension("OES_texture_float");

    this.textureCanvas = document.createElement("canvas");
    const ctx = this.textureCanvas.getContext("2d");
    if (!ctx) throw new Error("2D context not supported");
    this.textureCtx = ctx;

    this.initShaders();
    this.initBuffers();
    this.createTextCanvasTexture();
    this.resize();
  }

  private createShader(source: string, type: number): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error: " + gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  private createShaderProgram(
    vert: WebGLShader,
    frag: WebGLShader,
  ): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram()!;
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(program));
    }
    return program;
  }

  private getUniforms(
    program: WebGLProgram,
  ): Record<string, WebGLUniformLocation> {
    const gl = this.gl;
    const uniforms: Record<string, WebGLUniformLocation> = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(program, i)!.name;
      uniforms[name] = gl.getUniformLocation(program, name)!;
    }
    return uniforms;
  }

  private makeProgram(fragSrc: string): ShaderProgram {
    const vert = this.createShader(vertShader, this.gl.VERTEX_SHADER);
    const frag = this.createShader(fragSrc, this.gl.FRAGMENT_SHADER);
    const program = this.createShaderProgram(vert, frag);
    return { program, uniforms: this.getUniforms(program) };
  }

  private initShaders() {
    this.splatProgram = this.makeProgram(fragShaderPoint);
    this.divergenceProgram = this.makeProgram(fragShaderDivergence);
    this.pressureProgram = this.makeProgram(fragShaderPressure);
    this.gradientSubtractProgram = this.makeProgram(fragShaderGradientSubtract);
    this.advectionProgram = this.makeProgram(fragShaderAdvection);
    this.outputShaderProgram = this.makeProgram(fragShaderOutputShader);
  }

  private initBuffers() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
      gl.STATIC_DRAW,
    );
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(
      gl.ELEMENT_ARRAY_BUFFER,
      new Uint16Array([0, 1, 2, 0, 2, 3]),
      gl.STATIC_DRAW,
    );
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  private createTextCanvasTexture() {
    const gl = this.gl;
    this.canvasTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.canvasTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  updateTextCanvas() {
    const { textureCtx: ctx, textureCanvas: tc } = this;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, tc.width, tc.height);
    ctx.font =
      (this.params.isBold ? "bold" : "normal") +
      " " +
      this.params.fontSize * devicePixelRatio +
      "px " +
      fontOptions[this.params.fontName];
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.filter = "blur(3px)";
    const box = ctx.measureText(this.params.text);
    ctx.fillText(
      this.params.text,
      0.5 * tc.width,
      0.5 * tc.height + 0.5 * box.actualBoundingBoxAscent,
    );

    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.canvasTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tc);
  }

  private createFBO(w: number, h: number, type: number = this.gl.RGBA): FBO {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, type, w, h, 0, type, gl.FLOAT, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      fbo,
      width: w,
      height: h,
      attach(id: number) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      },
    };
  }

  private createDoubleFBO(w: number, h: number, type?: number): DoubleFBO {
    let fbo1 = this.createFBO(w, h, type);
    let fbo2 = this.createFBO(w, h, type);
    return {
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      read: () => fbo1,
      write: () => fbo2,
      swap() {
        const tmp = fbo1;
        fbo1 = fbo2;
        fbo2 = tmp;
      },
    };
  }

  private initFBOs() {
    const gl = this.gl;
    const w = Math.floor(0.5 * window.innerWidth);
    const h = Math.floor(0.5 * window.innerHeight);
    this.outputColor = this.createDoubleFBO(w, h);
    this.velocity = this.createDoubleFBO(w, h, gl.RGBA);
    this.divergence = this.createFBO(w, h, gl.RGB);
    this.pressure = this.createDoubleFBO(w, h, gl.RGB);
  }

  resize() {
    const { canvas, textureCanvas } = this;
    this.params.pointerSize = 4 / window.innerHeight;
    canvas.width = textureCanvas.width = window.innerWidth;
    canvas.height = textureCanvas.height = window.innerHeight;
    this.initFBOs();
    this.updateTextCanvas();
  }

  private blit(target: FBO | null) {
    const gl = this.gl;
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  updatePointer(eX: number, eY: number) {
    this.pointer.moved = true;
    this.pointer.dx = 5 * (eX - this.pointer.x);
    this.pointer.dy = 5 * (eY - this.pointer.y);
    this.pointer.x = eX;
    this.pointer.y = eY;
  }

  setUserInteracted() {
    this.isPreview = false;
  }

  private step(t: number) {
    const gl = this.gl;
    const { canvas, pointer, params } = this;
    const {
      splatProgram,
      divergenceProgram,
      pressureProgram,
      gradientSubtractProgram,
      advectionProgram,
      outputShaderProgram,
    } = this;
    const { outputColor, velocity, divergence, pressure } = this;
    const dt = 1 / 60;

    if (this.isPreview) {
      this.updatePointer(
        (0.5 - 0.45 * Math.sin(0.003 * t - 2)) * window.innerWidth,
        (0.5 + 0.1 * Math.sin(0.0025 * t) + 0.1 * Math.cos(0.002 * t)) *
          window.innerHeight,
      );
    }

    if (pointer.moved) {
      if (!this.isPreview) pointer.moved = false;

      gl.useProgram(splatProgram.program);
      gl.uniform1i(
        splatProgram.uniforms.u_input_texture,
        velocity.read().attach(1),
      );
      gl.uniform1f(splatProgram.uniforms.u_ratio, canvas.width / canvas.height);
      gl.uniform2f(
        splatProgram.uniforms.u_point,
        pointer.x / canvas.width,
        1 - pointer.y / canvas.height,
      );
      gl.uniform3f(
        splatProgram.uniforms.u_point_value,
        pointer.dx,
        -pointer.dy,
        1,
      );
      gl.uniform1f(splatProgram.uniforms.u_point_size, params.pointerSize);
      this.blit(velocity.write());
      velocity.swap();

      gl.uniform1i(
        splatProgram.uniforms.u_input_texture,
        outputColor.read().attach(1),
      );
      gl.uniform3f(
        splatProgram.uniforms.u_point_value,
        1 - params.color.r,
        1 - params.color.g,
        1 - params.color.b,
      );
      this.blit(outputColor.write());
      outputColor.swap();
    }

    gl.useProgram(divergenceProgram.program);
    gl.uniform2f(
      divergenceProgram.uniforms.u_texel,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(
      divergenceProgram.uniforms.u_velocity_texture,
      velocity.read().attach(1),
    );
    this.blit(divergence);

    gl.useProgram(pressureProgram.program);
    gl.uniform2f(
      pressureProgram.uniforms.u_texel,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(
      pressureProgram.uniforms.u_divergence_texture,
      divergence.attach(1),
    );
    for (let i = 0; i < 10; i++) {
      gl.uniform1i(
        pressureProgram.uniforms.u_pressure_texture,
        pressure.read().attach(2),
      );
      this.blit(pressure.write());
      pressure.swap();
    }

    gl.useProgram(gradientSubtractProgram.program);
    gl.uniform2f(
      gradientSubtractProgram.uniforms.u_texel,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(
      gradientSubtractProgram.uniforms.u_pressure_texture,
      pressure.read().attach(1),
    );
    gl.uniform1i(
      gradientSubtractProgram.uniforms.u_velocity_texture,
      velocity.read().attach(2),
    );
    this.blit(velocity.write());
    velocity.swap();

    gl.useProgram(advectionProgram.program);
    gl.uniform1f(advectionProgram.uniforms.u_use_text, 0);
    gl.uniform2f(
      advectionProgram.uniforms.u_texel,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(
      advectionProgram.uniforms.u_velocity_texture,
      velocity.read().attach(1),
    );
    gl.uniform1i(
      advectionProgram.uniforms.u_input_texture,
      velocity.read().attach(1),
    );
    gl.uniform1f(advectionProgram.uniforms.u_dt, dt);
    this.blit(velocity.write());
    velocity.swap();

    gl.useProgram(advectionProgram.program);
    gl.uniform1f(advectionProgram.uniforms.u_use_text, 1);
    gl.uniform2f(
      advectionProgram.uniforms.u_texel,
      outputColor.texelSizeX,
      outputColor.texelSizeY,
    );
    gl.uniform1i(
      advectionProgram.uniforms.u_input_texture,
      outputColor.read().attach(2),
    );
    this.blit(outputColor.write());
    outputColor.swap();

    gl.useProgram(outputShaderProgram.program);
    gl.uniform1i(
      outputShaderProgram.uniforms.u_output_texture,
      outputColor.read().attach(1),
    );
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  start() {
    const loop = (t: number) => {
      this.step(t);
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop() {
    cancelAnimationFrame(this.rafId);
  }
}
