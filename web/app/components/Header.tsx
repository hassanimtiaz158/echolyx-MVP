"use client";

import Logo from "@/app/components/Logo";

export default function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-[var(--border-soft)] bg-[var(--bg)]/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <Logo />
        <nav className="flex items-center gap-6">
          <a
            href="#demo"
            className="hidden text-sm font-medium text-[var(--text-dim)] transition-colors hover:text-[var(--text)] sm:block"
          >
            How it works
          </a>
          <a
            href="#demo"
            className="rounded-full bg-[var(--accent)] px-5 py-2 font-display text-sm font-semibold tracking-wide text-[#03181c] shadow-[0_0_20px_var(--accent-glow)] transition-transform hover:scale-105"
          >
            DEMO
          </a>
        </nav>
      </div>
    </header>
  );
}
