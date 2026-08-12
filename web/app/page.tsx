"use client";

import { useCallback, useRef, useState } from "react";
import ResultPanel from "@/app/components/ResultPanel";
import { ApiError, classifyAudio, type Prediction } from "@/lib/api";

type Status = "idle" | "recording" | "analyzing" | "done" | "error";

export default function Home() {
  const [clip, setClip] = useState<{ blob: Blob; name: string; url: string } | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetClip = useCallback((blob: Blob, name: string) => {
    setPrediction(null);
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
      const result = await classifyAudio(clip.blob, clip.name);
      setPrediction(result);
      setStatus("done");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing that clip.");
      setStatus("error");
    }
  }, [clip]);

  const isRecording = status === "recording";
  const isAnalyzing = status === "analyzing";

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden">
      {/* Ambient background */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-1/2 top-[-10%] h-[36rem] w-[36rem] -translate-x-1/2 rounded-full bg-sky-500/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-5%] h-[28rem] w-[28rem] rounded-full bg-indigo-500/10 blur-[120px]" />
      </div>

      <header className="mx-auto w-full max-w-3xl px-6 pt-14 pb-8 text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-500/15 ring-1 ring-sky-400/30">
          <span className="text-2xl">🌀</span>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
          Echolyx AI
        </h1>
        <p className="mt-2 text-balance text-slate-400">
          Point a microphone at a fan — get an instant{" "}
          <span className="text-emerald-300">Normal</span> vs{" "}
          <span className="text-rose-300">Faulty</span> read.
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Proof-of-concept demo · PANNs CNN14 transfer learning · not a production monitoring system
        </p>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-6 pb-16">
        <div className="grid gap-6 sm:grid-cols-2">
          {/* Upload / drop zone */}
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
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
              dragOver ? "border-sky-400 bg-sky-400/5" : "border-white/15 hover:border-white/25 hover:bg-white/5"
            }`}
          >
            <span className="text-3xl">📁</span>
            <p className="text-sm font-medium text-slate-200">Upload a clip</p>
            <p className="text-xs text-slate-500">wav / mp3 · drag &amp; drop or click</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => onFilePicked(e.target.files?.[0] ?? null)}
            />
          </div>

          {/* Record */}
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex flex-col items-center justify-center gap-2 rounded-2xl border-2 p-8 text-center transition-colors ${
              isRecording
                ? "border-rose-400/60 bg-rose-400/10"
                : "border-dashed border-white/15 hover:border-white/25 hover:bg-white/5"
            }`}
          >
            <span className={`text-3xl ${isRecording ? "animate-pulse" : ""}`}>
              {isRecording ? "⏹️" : "🎙️"}
            </span>
            <p className="text-sm font-medium text-slate-200">
              {isRecording ? "Stop recording" : "Record from mic"}
            </p>
            <p className="text-xs text-slate-500">~10s of fan sound works best</p>
          </button>
        </div>

        {clip && (
          <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <span className="text-xl">🎧</span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-200">{clip.name}</p>
                  <audio src={clip.url} controls className="mt-1 h-8 w-64 max-w-full" />
                </div>
              </div>
              <button
                type="button"
                onClick={analyze}
                disabled={isAnalyzing}
                className="shrink-0 rounded-xl bg-sky-500 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-sky-500/20 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isAnalyzing ? "Analyzing…" : "Analyze"}
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        {prediction && (
          <div className="mt-6">
            <ResultPanel prediction={prediction} />
          </div>
        )}
      </main>

      <footer className="mx-auto w-full max-w-3xl px-6 pb-10 text-center text-xs leading-relaxed text-slate-600">
        Research prototype, not a production monitoring system. Trained on the MIMII fan dataset;
        results may not generalize to arbitrary fans or machinery. Binary classification only —
        no fault localization.
      </footer>
    </div>
  );
}
