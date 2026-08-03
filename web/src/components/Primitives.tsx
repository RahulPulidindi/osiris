import type { ReactNode } from "react";

/** Small shared pieces. The design is mostly typography and hairlines, so the
 *  primitives stay deliberately thin. */

export function Label({ children }: { children: ReactNode }) {
  return <span className="label">{children}</span>;
}

/** Signed tone class from a number. */
export function tone(value: number): "up" | "down" | "flat" {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return "flat";
  return value > 0 ? "up" : "down";
}

/** Status word with its dot. The dot carries the state; the word explains it. */
export function StatusWord({
  kind,
  children,
}: {
  kind: "live" | "halt" | "warn" | "idle";
  children: ReactNode;
}) {
  const dotClass =
    kind === "live"
      ? "dot dot-live"
      : kind === "halt"
        ? "dot dot-halt"
        : kind === "warn"
          ? "dot dot-warn"
          : "dot";
  return (
    <span className="flex items-center gap-2">
      <span className={dotClass} />
      <span className="label" style={{ color: "var(--color-fg-2)" }}>
        {children}
      </span>
    </span>
  );
}

/** Empty state: one quiet sentence, never a fake chart. */
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="py-10">
      <p className="text-base text-[color:var(--color-fg-2)]">{title}</p>
      {children && (
        <p className="mt-2 max-w-md text-sm leading-relaxed text-[color:var(--color-fg-3)]">
          {children}
        </p>
      )}
    </div>
  );
}
