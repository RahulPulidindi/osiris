/** The Osiris mark: the Eye of Horus reduced to instrument geometry.
 *
 *  A thin outer ring (the dial), a horizon line through the middle (the
 *  market), and a small phosphor pupil sitting just above it (the agent,
 *  watching). One accent color, hairline strokes — the same vocabulary as the
 *  rest of the page, so it reads as an instrument, not a mascot. */
export function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      {/* Dial ring, broken at the lower right like a gauge that starts at open. */}
      <circle
        cx="16"
        cy="16"
        r="13.5"
        stroke="var(--color-fg-3)"
        strokeWidth="1.25"
        strokeDasharray="70 15"
        strokeDashoffset="-8"
        strokeLinecap="round"
      />
      {/* Horizon. */}
      <line
        x1="6.5"
        y1="19"
        x2="25.5"
        y2="19"
        stroke="var(--color-fg-4)"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
      {/* The eye: an arc over the horizon… */}
      <path
        d="M9.5 19c1.8-4.2 4-6.3 6.5-6.3s4.7 2.1 6.5 6.3"
        stroke="var(--color-fg)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      {/* …and the phosphor pupil. */}
      <circle cx="16" cy="16.4" r="2.1" fill="var(--color-up)" />
    </svg>
  );
}
