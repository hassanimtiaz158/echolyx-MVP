"use client";

import { useCallback, useRef, useState } from "react";
import Header from "@/app/components/Header";
import DiagnosticsPanel from "@/app/components/DiagnosticsPanel";
import { ApiError, classifyAudio, type Prediction } from "@/lib/api";
import { analyzeAudio, decodeAudio, type AudioMetrics } from "@/lib/audioAnalysis";

type Status = "idle" | "recording" | "analyzing" | "done" | "error";

export default function Home() {
  const [clip, setClip] = useState<{ blob: Blob; name: string; url: string } | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [metrics, setMetrics] = useState<AudioMetrics | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetClip = useCallback((blob: Blob, name: string) => {
    setPrediction(null);
    setMetrics(null);
    setError(null);
    setStatus("idle");
    setClip((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return { blob, name, url: URL.createObjectURL(blob) };
    });
  }, []);

  const onFilePicked = useCallback(
    (file: File | null) => {
      if (!file) return;
      resetClip(file, file.name);
    },
    [resetClip]
  );

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        resetClip(blob, "recording.webm");
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setStatus("recording");
    } catch {
      setError("Microphone access was denied or is unavailable.");
      setStatus("error");
    }
  }, [resetClip]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    setStatus("idle");
  }, []);

  const analyze = useCallback(async () => {
    if (!clip) return;
    setStatus("analyzing");
    setError(null);
    try {
      const [result, decoded] = await Promise.all([
        classifyAudio(clip.blob, clip.name),
        decodeAudio(clip.blob),
      ]);
      setPrediction(result);
      setMetrics(analyzeAudio(decoded.samples, decoded.sampleRate));
      setStatus("done");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing that clip.");
      setStatus("error");
    }
  }, [clip]);

  const isRecording = status === "recording";
  const isAnalyzing = status === "analyzing";

  return (
    <div className="flex flex-1 flex-col">
      <Header />

      {/* Hero */}
      <section className="relative overflow-hidden border-b border-[var(--border-soft)]">
        <div
          className="pointer-events-none absolute inset-0 -z-10"
          style={{
            background:
              "radial-gradient(600px 300px at 50% -10%, var(--accent-soft), transparent 70%)",
          }}
        />
        <div className="mx-auto max-w-3xl px-6 py-20 text-center">
          <p className="font-display text-xs font-semibold tracking-[0.3em] text-[var(--accent)]">
            ACOUSTIC FAULT DIAGNOSTICS
          </p>
          <h1 className="mt-4 text-balance font-display text-4xl font-bold leading-tight text-[var(--text)] sm:text-5xl">
            Hear a fault before it
            <br />
            <span style={{ color: "var(--accent)" }}>becomes a failure.</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-balance text-[var(--text-dim)]">
            Upload or record ~10 seconds of fan audio. A PANNs CNN14 transfer-learning model,
            fine-tuned on real industrial fan recordings, returns an instant diagnostic —
            waveform, spectrogram, and anomaly score included.
          </p>
          <a
            href="#demo"
            className="mt-8 inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-7 py-3 font-display text-sm font-semibold tracking-wide text-[#03181c] shadow-[0_0_28px_var(--accent-glow)] transition-transform hover:scale-105"
          >
            LAUNCH DEMO
          </a>
          <p className="mt-4 text-xs text-[var(--text-faint)]">
            Proof-of-concept — not a production monitoring system.
          </p>
        </div>
      </section>

      {/* Demo / diagnostics console */}
      <main id="demo" className="mx-auto w-full max-w-3xl flex-1 px-6 py-14">
        <p className="font-display text-xs font-semibold tracking-[0.14em] text-[var(--text-dim)]">
          DIAGNOSTIC CONSOLE
        </p>
        <h2 className="mt-1 font-display text-2xl font-bold text-[var(--text)]">Analyze Fan Audio</h2>
        <p className="mt-1 text-sm text-[var(--text-dim)]">Upload a clip or record from your microphone.</p>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              onFilePicked(e.dataTransfer.files?.[0] ?? null);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-8 text-center transition-colors"
            style={{
              borderColor: dragOver ? "var(--accent)" : "var(--border)",
              background: dragOver ? "var(--accent-soft)" : "var(--surface)",
            }}
          >
            <span className="text-3xl" aria-hidden="true">
              ↑
            </span>
            <p className="font-display text-sm font-semibold text-[var(--text)]">
              Drop audio file here or click to browse
            </p>
            <p className="text-xs text-[var(--text-dim)]">Supports WAV, MP3, FLAC · 5–30 seconds</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => onFilePicked(e.target.files?.[0] ?? null)}
            />
          </div>

          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            className="flex flex-col items-center justify-center gap-2 rounded-2xl border-2 p-8 text-center transition-colors"
            style={{
              borderColor: isRecording ? "var(--critical)" : "var(--border)",
              borderStyle: isRecording ? "solid" : "dashed",
              background: isRecording ? "var(--critical-soft)" : "var(--surface)",
            }}
          >
            <span className={`text-3xl ${isRecording ? "animate-pulse" : ""}`} aria-hidden="true">
              {isRecording ? "■" : "●"}
            </span>
            <p className="font-display text-sm font-semibold text-[var(--text)]">
              {isRecording ? "Stop recording" : "Record from mic"}
            </p>
            <p className="text-xs text-[var(--text-dim)]">~10s of fan sound works best</p>
          </button>
        </div>

        {clip && (
          <div className="mt-5 rounded-2xl border border-[var(--border-soft)] bg-[var(--surface)] p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <span
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm"
                  style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                >
                  ♪
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-[var(--text)]">{clip.name}</p>
                  <audio src={clip.url} controls className="mt-1 h-8 w-64 max-w-full" />
                </div>
              </div>
              <button
                type="button"
                onClick={analyze}
                disabled={isAnalyzing}
                className="shrink-0 rounded-full bg-[var(--accent)] px-6 py-2.5 font-display text-sm font-semibold tracking-wide text-[#03181c] shadow-[0_0_20px_var(--accent-glow)] transition-transform hover:scale-105 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
              >
                {isAnalyzing ? "Analyzing…" : "Analyze"}
              </button>
            </div>
          </div>
        )}

        {error && (
          <div
            className="mt-5 rounded-xl border px-4 py-3 text-sm"
            style={{ borderColor: "var(--critical)", background: "var(--critical-soft)", color: "#ffb4c2" }}
          >
            {error}
          </div>
        )}

        {prediction && metrics && (
          <div className="mt-6">
            <DiagnosticsPanel prediction={prediction} metrics={metrics} />
          </div>
        )}
      </main>

      <footer className="mx-auto w-full max-w-3xl px-6 pb-10 text-center text-xs leading-relaxed text-[var(--text-faint)]">
        Research prototype, not a production monitoring system. Trained on the MIMII fan dataset;
        results may not generalize to arbitrary fans or machinery. Binary classification only —
        no fault localization.
      </footer>
    </div>
  );
}
