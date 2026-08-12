"use client";

import type { AudioMetrics } from "@/lib/audioAnalysis";
import type { Prediction } from "@/lib/api";
import Waveform from "@/app/components/Waveform";
import Spectrogram from "@/app/components/Spectrogram";

const STATUS_MAP: Record<string, { label: string; color: string; soft: string }> = {
  Normal: { label: "HEALTHY", color: "var(--healthy)", soft: "var(--healthy-soft)" },
  Faulty: { label: "CRITICAL", color: "var(--critical)", soft: "var(--critical-soft)" },
  Uncertain: { label: "STRESSED", color: "var(--stressed)", soft: "var(--stressed-soft)" },
};

function Metric({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="rounded-xl border border-[var(--border-soft)] bg-[var(--surface)] p-4">
      <p className="font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
        {label}
      </p>
      <p className="tabular mt-1.5 font-display text-2xl font-semibold text-[var(--text)]">
        {value} <span className="text-sm font-medium text-[var(--text-dim)]">{unit}</span>
      </p>
    </div>
  );
}

function insightLines(prediction: Prediction, metrics: AudioMetrics): string[] {
  const pFaulty = prediction.probabilities["Faulty"] ?? 0;
  const lines: string[] = [
    `${prediction.label} classification at ${(prediction.confidence * 100).toFixed(1)}% confidence (PANNs CNN14 transfer-learning model).`,
  ];

  if (metrics.instabilityIndex > 3) {
    lines.push(
      `Elevated amplitude variance (instability index ${metrics.instabilityIndex.toFixed(2)}) — envelope is irregular across the clip, consistent with rattling or intermittent contact.`
    );
  } else {
    lines.push(
      `Amplitude envelope is stable (instability index ${metrics.instabilityIndex.toFixed(2)}) — no strong sign of intermittent mechanical contact.`
    );
  }

  if (pFaulty >= 0.5) {
    lines.push("Recommend a follow-up recording closer to the housing to confirm before servicing.");
  } else if (pFaulty >= 0.05) {
    lines.push("Borderline reading — a second, longer recording would sharpen this result.");
  } else {
    lines.push("No corrective action indicated by this reading.");
  }

  return lines;
}

export default function DiagnosticsPanel({
  prediction,
  metrics,
}: {
  prediction: Prediction;
  metrics: AudioMetrics;
}) {
  const status = STATUS_MAP[prediction.label] ?? STATUS_MAP.Uncertain;
  const anomalyScore = (prediction.probabilities["Faulty"] ?? 0) * 100;

  return (
    <div
      className="rounded-2xl border p-6"
      style={{ borderColor: status.color + "40", background: "var(--bg-raised)" }}
    >
      <div className="grid grid-cols-2 gap-6">
        <div>
          <p className="font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
            DIAGNOSTIC STATUS
          </p>
          <div className="mt-2 flex items-center gap-2.5">
            <span
              className="h-3 w-3 rounded-full"
              style={{ background: status.color, boxShadow: `0 0 12px ${status.color}` }}
            />
            <span
              className="font-display text-3xl font-bold tracking-wide"
              style={{ color: status.color }}
            >
              {status.label}
            </span>
          </div>
        </div>
        <div className="text-right">
          <p className="font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
            ANOMALY SCORE MATRIX
          </p>
          <p
            className="tabular mt-2 font-display text-4xl font-bold"
            style={{ color: status.color }}
          >
            {anomalyScore.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="PEAK FREQUENCY" value={metrics.peakFrequencyHz.toFixed(0)} unit="Hz" />
        <Metric label="SIGNAL STRENGTH" value={metrics.rms.toFixed(2)} unit="RMS" />
        <Metric label="SPECTRAL ROLLOFF" value={metrics.spectralRolloffHz.toFixed(0)} unit="Hz" />
        <Metric label="INSTABILITY INDEX" value={metrics.instabilityIndex.toFixed(2)} unit="STD" />
      </div>

      <div className="mt-6">
        <div className="mb-2 flex items-baseline justify-between">
          <p className="font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
            AUDIO WAVEFORM
          </p>
          <p className="tabular text-xs font-medium" style={{ color: status.color }}>
            {metrics.durationSec.toFixed(1)}S · {status.label}
          </p>
        </div>
        <div className="rounded-lg border border-[var(--border-soft)] bg-black/30 p-2">
          <Waveform points={metrics.waveform} color={status.color} />
        </div>
      </div>

      <div className="mt-6">
        <p className="mb-2 font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
          SPECTRAL SPECTROGRAM
        </p>
        <Spectrogram matrix={metrics.spectrogram} />
      </div>

      <div className="mt-6 rounded-xl border border-[var(--border-soft)] bg-[var(--surface)] px-4 py-3">
        <p className="mb-2.5 font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
          DIAGNOSTICS LEGEND
        </p>
        <div className="flex flex-wrap gap-5 text-xs text-[var(--text-dim)]">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: "var(--healthy)" }} />
            HEALTHY
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: "var(--stressed)" }} />
            STRESSED
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: "var(--critical)" }} />
            CRITICAL
          </span>
        </div>
      </div>

      <div
        className="mt-4 rounded-xl border-l-2 bg-[var(--surface)] px-4 py-3.5"
        style={{ borderColor: status.color }}
      >
        <div className="flex items-start gap-3">
          <span
            className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
            style={{ background: status.soft, color: status.color }}
            aria-hidden="true"
          >
            ⚡
          </span>
          <div>
            <p className="font-display text-sm font-semibold text-[var(--text)]">Inference Insight</p>
            <ul className="mt-1.5 space-y-1 text-sm leading-relaxed text-[var(--text-dim)]">
              {insightLines(prediction, metrics).map((line, i) => (
                <li key={i}>· {line}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
