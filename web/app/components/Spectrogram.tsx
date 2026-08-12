"use client";

import { useEffect, useRef } from "react";

// A compact magma-like colormap: near-black -> violet -> orange -> pale yellow.
const STOPS: [number, number, number, number][] = [
  [0, 6, 4, 20],
  [0.25, 68, 15, 92],
  [0.5, 152, 32, 92],
  [0.7, 222, 73, 60],
  [0.85, 250, 152, 55],
  [1, 252, 232, 150],
];

function colormap(t: number): [number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t0, r0, g0, b0] = STOPS[i - 1];
    const [t1, r1, g1, b1] = STOPS[i];
    if (c <= t1) {
      const f = (c - t0) / (t1 - t0 || 1);
      return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f];
    }
  }
  return [252, 232, 150];
}

export default function Spectrogram({
  matrix,
  height = 150,
}: {
  matrix: number[][];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || matrix.length === 0) return;

    const nFrames = matrix.length;
    const nBins = matrix[0].length;
    // Only render the lower ~40% of bins (fan fault energy lives well below
    // Nyquist; the upper bins are mostly empty and just waste pixels).
    const visibleBins = Math.max(8, Math.floor(nBins * 0.4));

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = container.clientWidth;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);

      const off = document.createElement("canvas");
      off.width = nFrames;
      off.height = visibleBins;
      const octx = off.getContext("2d")!;
      const img = octx.createImageData(nFrames, visibleBins);
      for (let f = 0; f < nFrames; f++) {
        for (let b = 0; b < visibleBins; b++) {
          const v = matrix[f][b] ?? 0;
          const [r, g, bl] = colormap(v);
          const row = visibleBins - 1 - b; // low freq at bottom
          const idx = (row * nFrames + f) * 4;
          img.data[idx] = r;
          img.data[idx + 1] = g;
          img.data[idx + 2] = bl;
          img.data[idx + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, nFrames, visibleBins, 0, 0, width, height);
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [matrix, height]);

  return (
    <div ref={containerRef} className="w-full overflow-hidden rounded-lg">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  );
}
