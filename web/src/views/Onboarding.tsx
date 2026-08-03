import { useState } from "react";
import { api } from "../lib/api";
import { useStore } from "../store/useStore";

/** Two steps before the dashboard exists: link the account, accept the risk.
 *
 *  Linking is a terminal command by design — OAuth needs a browser consent and
 *  the token cache lives outside this repo, so the page's job is to say the
 *  command clearly and notice when it has been done (the poll picks it up).
 *
 *  The acknowledgement is typed, not a checkbox. It writes both affirmations
 *  to .env; the service restart is what actually arms the account, so this
 *  page can never arm live trading in one click. */
export function Onboarding() {
  const info = useStore((s) => s.connectionInfo);
  if (!info) return null;

  const step = !info.robinhood_linked ? 1 : 2;

  return (
    <div className="mx-auto max-w-lg py-14">
      <div className="rise">
        <p className="label">Setup · step {step} of 2</p>
        <h1 className="mt-3 text-2xl font-medium tracking-tight">
          {step === 1 ? "Connect your Robinhood account" : "Understand the risk"}
        </h1>
      </div>

      <div className="rise mt-8" style={{ animationDelay: "80ms" }}>
        {step === 1 ? <LinkStep /> : <RiskStep restart={info.restart_required} />}
      </div>

      <ol className="rise mt-12 flex gap-2" style={{ animationDelay: "160ms" }}>
        {[1, 2].map((n) => (
          <li
            key={n}
            className="h-1 flex-1 rounded-full"
            style={{
              background:
                n < step
                  ? "var(--color-up-dim)"
                  : n === step
                    ? "var(--color-fg-3)"
                    : "var(--color-line)",
            }}
          />
        ))}
      </ol>
    </div>
  );
}

function LinkStep() {
  return (
    <div className="text-sm leading-relaxed text-[color:var(--color-fg-2)]">
      <p>
        Robinhood&rsquo;s consent screen opens in your browser once. Run this in
        a terminal at the project root:
      </p>
      <code className="cmd mt-4">python -m osiris.connect</code>
      <p className="mt-4 text-[color:var(--color-fg-3)]">
        It signs in, reads your balance and positions back to you, and caches
        credentials outside this repository. This page updates on its own once
        the account is readable.
      </p>
    </div>
  );
}

function RiskStep({ restart }: { restart: boolean }) {
  const [phrase, setPhrase] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(restart);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const info = await api.acknowledgeRisk(phrase.trim(), name.trim());
      useStore.setState({ connectionInfo: info });
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <div className="text-sm leading-relaxed text-[color:var(--color-fg-2)]">
        <p>
          Acknowledged and recorded. One deliberate step remains: restart the
          service so the live connection opens with trading enabled.
        </p>
        <code className="cmd mt-4">python -m osiris.run --serve</code>
        <p className="mt-4 text-[color:var(--color-fg-3)]">
          This page cannot arm the account by itself — the restart is the
          second factor, on purpose.
        </p>
      </div>
    );
  }

  return (
    <div className="text-sm leading-relaxed text-[color:var(--color-fg-2)]">
      <p>
        Osiris will buy and sell in your account on its own, without asking
        first. It caps position sizes and halts on losses, but it can and will
        lose real money. Most retail trading underperforms simply holding an
        index fund.
      </p>
      <p className="mt-3 text-[color:var(--color-fg-3)]">
        A stop button is always in the header, and{" "}
        <span className="num text-xs">touch KILL_SWITCH</span> stops it even if
        this page will not load.
      </p>

      <div className="mt-6 flex flex-col gap-3">
        <input
          className="field"
          placeholder='Type "I understand the risk"'
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          autoComplete="off"
        />
        <input
          className="field"
          placeholder="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoComplete="name"
        />
        <button
          type="button"
          className="btn btn-solid mt-1 self-start"
          disabled={
            busy ||
            phrase.trim().toLowerCase() !== "i understand the risk" ||
            !name.trim()
          }
          onClick={submit}
        >
          {busy ? "Recording…" : "Enable live trading"}
        </button>
        {error && <p className="text-[color:var(--color-down)]">{error}</p>}
      </div>
    </div>
  );
}
