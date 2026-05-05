import { useEffect, useRef } from "react";
import GUI from "lil-gui";
import { FluidSimulation, fontOptions } from "./fluid";

export default function FluidCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const sim = new FluidSimulation(canvas);

    const gui = new GUI();
    gui.close();
    gui.add(sim.params, "text").onChange(() => sim.updateTextCanvas());
    gui.add(sim.params, "fontSize", 10, 300).name("font size, px").onChange(() => sim.updateTextCanvas());
    gui.add(sim.params, "isBold").name("bold").onChange(() => sim.updateTextCanvas());
    gui.add(sim.params, "fontName", Object.keys(fontOptions)).name("font").onChange(() => sim.updateTextCanvas());
    gui.addColor(sim.params, "color");

    const onMouseMove = (e: MouseEvent) => {
      sim.setUserInteracted();
      sim.updatePointer(e.pageX, e.pageY);
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      sim.setUserInteracted();
      sim.updatePointer(e.targetTouches[0].pageX, e.targetTouches[0].pageY);
    };
    const onResize = () => sim.resize();

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("resize", onResize);

    sim.start();

    return () => {
      sim.stop();
      gui.destroy();
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return <canvas ref={canvasRef} />;
}
