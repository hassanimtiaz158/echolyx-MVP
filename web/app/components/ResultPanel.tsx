"use client";

import type { Prediction } from "@/lib/api";

const LABEL_STYLES: Record<string, { badge: string; ring: string; dot: string }> = {
  Normal: {
    badge: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
    ring: "ring-emerald-400/40",
    dot: "bg-emerald-400",
  },
  Faulty: {
    badge: "bg-rose-500/15 text-rose-300 ring-rose-400/30",
    ring: "ring-rose-400/40",
    dot: "bg-rose-400",
  },
  Uncertain: {
    badge: "bg-amber-500/15 text-amber-300 ring-amber-400/30",
    ring: "ring-amber-400/40",
    dot: "bg-amber-400",
  },
};

const BAR_COLORS: Record<string, string> = {
  Normal: "bg-emerald-400",
  Faulty: "bg-rose-400",
};

function fmtPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

export default function ResultPanel({ prediction }: { prediction: Prediction }) {
  const style = LABEL_STYLES[prediction.label] ?? LABEL_STYLES.Uncertain;
  const entries = Object.entries(prediction.probabilities);

  return (
    <div className={`rounded-2xl border border-white/10 bg-white/5 p-6 ring-1 ${style.ring} backdrop-blur`}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">Prediction</p>
          <div className={`mt-2 inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-lg font-semibold ring-1 ${style.badge}`}>
            <span className={`h-2 w-2 rounded-full ${style.dot}`} />
            {prediction.label}
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-widest text-slate-400">Confidence</p>
          <p className="mt-1 font-mono text-3xl font-semibold text-slate-100">
            {fmtPct(prediction.confidence)}
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-3">
        {entries.map(([name, prob]) => (
          <div key={name}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-slate-300">{name}</span>
              <span className="font-mono text-slate-400">{fmtPct(prob)}</span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${BAR_COLORS[name] ?? "bg-sky-400"}`}
                style={{ width: `${Math.max(prob * 100, 2)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {prediction.label === "Uncertain" && (
        <p className="mt-5 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90">
          The model isn&apos;t confident either way — try a clearer or longer recording.
        </p>
      )}
    </div>
  );
}
