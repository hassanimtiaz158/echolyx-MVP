// Real client-side DSP on the actual uploaded/recorded clip — no canned
// numbers. A small radix-2 FFT drives peak frequency, spectral rolloff, and
// a short-time spectrogram; RMS drives signal strength and an "instability
// index" (std-dev of framewise RMS, a cheap proxy for how erratic the
// amplitude envelope is — steady hum vs. rattling/grinding).

export type AudioMetrics = {
  durationSec: number;
  rms: number;
  peakFrequencyHz: number;
  spectralRolloffHz: number;
  instabilityIndex: number;
  waveform: number[]; // downsampled envelope, 0..1
  spectrogram: number[][]; // [frame][bin] magnitude, normalized 0..1
  binHz: number; // frequency resolution per spectrogram bin
};

export async function decodeAudio(source: Blob): Promise<{ samples: Float32Array; sampleRate: number }> {
  const arrayBuffer = await source.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const channels = decoded.numberOfChannels;
    const length = decoded.length;
    const mono = new Float32Array(length);
    for (let c = 0; c < channels; c++) {
      const data = decoded.getChannelData(c);
      for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
    }
    return { samples: mono, sampleRate: decoded.sampleRate };
  } finally {
    ctx.close();
  }
}

// Iterative in-place radix-2 Cooley-Tukey FFT. `re`/`im` length must be a power of 2.
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }
}

const FRAME_SIZE = 1024; // power of 2
const TARGET_FRAMES = 110;

function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

export function analyzeAudio(samples: Float32Array, sampleRate: number): AudioMetrics {
  const durationSec = samples.length / sampleRate;

  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
  const rms = Math.sqrt(sumSq / samples.length);

  const hop = Math.max(256, Math.floor(Math.max(1, samples.length - FRAME_SIZE) / TARGET_FRAMES));
  const window = hannWindow(FRAME_SIZE);
  const nBins = FRAME_SIZE / 2;
  const binHz = sampleRate / FRAME_SIZE;

  const spectrogram: number[][] = [];
  const frameRms: number[] = [];
  const avgMag = new Float64Array(nBins);
  let frameCount = 0;

  for (let start = 0; start + FRAME_SIZE <= samples.length; start += hop) {
    const re = new Float64Array(FRAME_SIZE);
    const im = new Float64Array(FRAME_SIZE);
    let frameSumSq = 0;
    for (let i = 0; i < FRAME_SIZE; i++) {
      const s = samples[start + i] * window[i];
      re[i] = s;
      frameSumSq += s * s;
    }
    fft(re, im);
    const mags = new Array<number>(nBins);
    for (let b = 0; b < nBins; b++) {
      const m = Math.hypot(re[b], im[b]);
      mags[b] = m;
      avgMag[b] += m;
    }
    spectrogram.push(mags);
    frameRms.push(Math.sqrt(frameSumSq / FRAME_SIZE));
    frameCount++;
  }

  for (let b = 0; b < nBins; b++) avgMag[b] /= Math.max(1, frameCount);

  // Peak frequency + 85% spectral rolloff from the time-averaged spectrum
  // (skip DC bin 0).
  let peakBin = 1;
  let totalEnergy = 0;
  for (let b = 1; b < nBins; b++) {
    totalEnergy += avgMag[b];
    if (avgMag[b] > avgMag[peakBin]) peakBin = b;
  }
  let cumEnergy = 0;
  let rolloffBin = peakBin;
  for (let b = 1; b < nBins; b++) {
    cumEnergy += avgMag[b];
    if (cumEnergy >= 0.85 * totalEnergy) {
      rolloffBin = b;
      break;
    }
  }

  const meanFrameRms = frameRms.reduce((a, b) => a + b, 0) / Math.max(1, frameRms.length);
  const variance =
    frameRms.reduce((a, v) => a + (v - meanFrameRms) ** 2, 0) / Math.max(1, frameRms.length);
  const instabilityIndex = Math.sqrt(variance) * 100;

  // Downsample raw samples to a compact envelope for the waveform plot.
  const targetPoints = 240;
  const bucket = Math.max(1, Math.floor(samples.length / targetPoints));
  const waveform: number[] = [];
  for (let i = 0; i < samples.length; i += bucket) {
    let peak = 0;
    for (let j = i; j < Math.min(i + bucket, samples.length); j++) {
      const a = Math.abs(samples[j]);
      if (a > peak) peak = a;
    }
    waveform.push(peak);
  }

  // Normalize spectrogram to 0..1 (log-scaled magnitude) for the colormap.
  let maxMag = 1e-6;
  for (const frame of spectrogram) for (const m of frame) if (m > maxMag) maxMag = m;
  const logMax = Math.log1p(maxMag);
  const normSpectrogram = spectrogram.map((frame) =>
    frame.map((m) => Math.log1p(m) / logMax)
  );

  return {
    durationSec,
    rms,
    peakFrequencyHz: peakBin * binHz,
    spectralRolloffHz: rolloffBin * binHz,
    instabilityIndex,
    waveform,
    spectrogram: normSpectrogram,
    binHz,
  };
}
