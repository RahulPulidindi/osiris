import { useEffect, useState } from "react";
import { Logo } from "./components/Logo";
import { StatusWord } from "./components/Primitives";
import { api } from "./lib/api";
import { useStore } from "./store/useStore";
import { Dashboard } from "./views/Dashboard";
import { Onboarding } from "./views/Onboarding";

/** One page. The portfolio is the page; positions and the agent's actions run
 *  down it in order of importance. Setup lives behind the same URL and simply
 *  replaces the page until the account is linked and acknowledged. */
export function App() {
  const refreshAll = useStore((s) => s.refreshAll);
  const connect = useStore((s) => s.connect);
  const connectionInfo = useStore((s) => s.connectionInfo);
  const health = useStore((s) => s.health);
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    void refreshAll().finally(() => setBooted(true));
    const disconnect = connect();
    // Slow poll as a safety net for a silently dead SSE stream.
    const interval = window.setInterval(() => void refreshAll(), 60_000);
    return () => {
      disconnect();
      window.clearInterval(interval);
    };
  }, [refreshAll, connect]);

  // Setup is shown only when we know the account is not linked or the risk is
  // unacknowledged. While the first fetch is in flight, show nothing rather
  // than flashing the wrong screen.
  const needsSetup =
    connectionInfo != null &&
    (!connectionInfo.robinhood_linked || !connectionInfo.risk_acknowledged);

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-[1360px] flex-col px-6 sm:px-10 lg:px-14">
      <Header />
      <main className="flex-1 pb-16">
        {!booted ? null : needsSetup ? <Onboarding /> : <Dashboard />}
      </main>
      <Footer apiDown={booted && health == null} />
    </div>
  );
}

function Header() {
  const health = useStore((s) => s.health);
  const [busy, setBusy] = useState(false);

  const stopped = health?.kill_switch_engaged ?? false;
  const halted = (health?.breakers_tripped.length ?? 0) > 0;

  async function toggle() {
    if (stopped) {
      const who = window.prompt(
        "Your name — this lets the agent open new positions again:",
      );
      if (!who?.trim()) return;
      setBusy(true);
      try {
        useStore.setState({ health: await api.releaseKillSwitch(who.trim()) });
      } finally {
        setBusy(false);
      }
    } else {
      const reason = window.prompt(
        "Why are you stopping the agent? It will still manage and exit what it holds.",
      );
      if (!reason?.trim()) return;
      setBusy(true);
      try {
        useStore.setState({ health: await api.engageKillSwitch(reason.trim()) });
      } finally {
        setBusy(false);
      }
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-[color:var(--color-line-soft)] py-5">
      <span className="flex items-center gap-3">
        <Logo />
        <span className="wordmark">Osiris</span>
      </span>
      <div className="flex items-center gap-5">
        {health &&
          (stopped ? (
            <StatusWord kind="halt">Stopped</StatusWord>
          ) : halted ? (
            <StatusWord kind="halt">Halted</StatusWord>
          ) : health.armed ? (
            <StatusWord kind="live">Live</StatusWord>
          ) : health.mode === "live" ? (
            <StatusWord kind="warn">Not armed</StatusWord>
          ) : (
            <StatusWord kind="idle">Observing</StatusWord>
          ))}
        {health && (
          <button
            type="button"
            className={stopped ? "btn" : "btn btn-danger"}
            disabled={busy}
            onClick={toggle}
          >
            {stopped ? "Resume" : "Stop"}
          </button>
        )}
      </div>
    </header>
  );
}

function Footer({ apiDown }: { apiDown: boolean }) {
  const connection = useStore((s) => s.connection);
  const connectionInfo = useStore((s) => s.connectionInfo);
  const lastError = useStore((s) => s.lastError);

  return (
    <footer className="border-t border-[color:var(--color-line-soft)] py-4">
      <div className="flex items-center justify-between text-xs text-[color:var(--color-fg-3)]">
        <span>
          {apiDown
            ? "service offline"
            : connectionInfo?.robinhood_linked
              ? "Robinhood connected"
              : "Robinhood not connected"}
          {" · "}
          {connection === "open" ? "streaming" : connection}
        </span>
        {lastError && (
          <span className="text-[color:var(--color-down)]">{lastError}</span>
        )}
      </div>
    </footer>
  );
}
