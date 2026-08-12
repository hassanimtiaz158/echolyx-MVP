export default function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <svg width="22" height="18" viewBox="0 0 22 18" fill="none" aria-hidden="true">
        <path
          d="M1 9H5L7 3L11 15L13.5 9H16L17.5 5.5L19 9H21"
          stroke="var(--accent)"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="font-display text-lg font-semibold tracking-[0.18em] text-[var(--text)]">
        ECHOLYX
      </span>
    </div>
  );
}
