"use client";

import { useEffect, useRef } from "react";

export default function Waveform({
  points,
  color,
  height = 110,
}: {
  points: number[];
  color: string;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || points.length === 0) return;

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
      ctx.clearRect(0, 0, width, height);

      const mid = height / 2;
      const barGap = 1.5;
      const barWidth = Math.max(1, width / points.length - barGap);
      const maxPeak = Math.max(...points, 0.001);

      const topPath = new Path2D();
      points.forEach((p, i) => {
        const norm = p / maxPeak;
        const x = (i / points.length) * width;
        const barH = Math.max(1.5, norm * (height * 0.42));

        ctx.fillStyle = color + "33";
        ctx.fillRect(x, mid - barH, barWidth, barH * 2);

        if (i === 0) topPath.moveTo(x, mid - barH);
        else topPath.lineTo(x, mid - barH);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.stroke(topPath);
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [points, color, height]);

  return (
    <div ref={containerRef} className="w-full">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  );
}
