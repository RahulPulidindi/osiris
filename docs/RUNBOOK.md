# Osiris — Operations Runbook

For the person on the hook when something goes wrong, which will eventually be
you at an inconvenient hour. Procedures are ordered by what to do *first*, not by
what is most interesting.

The single rule that governs everything below: **halting is always safe, and
resuming never is.** When uncertain, halt. A missed day of trading costs nothing
you can measure; trading on a corrupt ledger costs the account.

---

## Commands

Run `source .venv/bin/activate` first, or substitute `.venv/bin/python` for
`python`. The system interpreter has none of the dependencies.

```bash
python -m osiris.connect              # authorize Robinhood (browser, once)
python -m osiris.connect --status     # is it connected?
python -m osiris.connect --logout     # forget credentials

python -m osiris.run --once --dry-run # decide + explain, place nothing
python -m osiris.run --once           # one real cycle
python -m osiris.run --serve          # dashboard + scheduled cycles
python -m osiris.run --unrestricted   # disable position/loss caps

python -m osiris.runner.gate          # offline strategy validation
```

Real orders additionally require `OSIRIS_MODE=live` **and**
`OSIRIS_I_UNDERSTAND_THE_RISK=yes`, set in `.env` rather than with `export`
(which dies with the shell).

---

## Running unattended

`--serve` wakes on the market's schedule: once per session about ten minutes
after the open, skipping weekends, holidays, pre-market, and the last twenty
minutes before the close. It prints that schedule at startup, so a process that
is sleeping looks different from one that has hung.

The agent needs no instruction to trade. There is no command to issue and no
button in the UI — the dashboard is read-only by design.

### Keeping it alive across reboots (macOS)

Save as `~/Library/LaunchAgents/com.osiris.agent.plist`, substituting your path:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.osiris.agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/Desktop/osiris/.venv/bin/python</string>
    <string>-m</string><string>osiris.run</string><string>--serve</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/Desktop/osiris</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/YOU/Desktop/osiris/data/agent.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/Desktop/osiris/data/agent.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.osiris.agent.plist    # start
launchctl unload ~/Library/LaunchAgents/com.osiris.agent.plist  # stop
tail -f data/agent.log                                          # watch
```

`KeepAlive` restarts the process if it crashes. Note that it will also restart it
after a deliberate `kill` — engage the kill switch instead, which survives a
restart precisely because it is a file on disk.

**A supervisor cannot un-halt the agent.** `KILL_SWITCH` is checked on every
action, so a restarted process comes back halted. That is the intended behaviour:
a supervisor that resurrects a halted agent into an un-halted state is a real
failure mode.

---

## 0. Emergency stop

```bash
touch /path/to/osiris/KILL_SWITCH
echo "reason for halt" > /path/to/osiris/KILL_SWITCH   # better: record why
```

Effective on the next action check. No process restart, no API, no dashboard
required — deliberately, because those are exactly what is unavailable during an
incident.

**What it does:** blocks all new entries.
**What it does *not* do:** stop risk exits or stop-losses. Halt means take no new
risk, never abandon existing risk. If you need to be fully flat, that is a
separate, manual decision made through the broker.

Release requires a named human acknowledgement — automated release defeats the
purpose:

```bash
curl -XPOST localhost:8000/api/control/kill-switch/release \
     -H 'content-type: application/json' -d '{"acknowledged_by":"your name"}'
```

---

## 1. Daily operation

| When | Action |
|---|---|
| Pre-open | `GET /api/preflight` — confirm cleared, read every advisory |
| Post-cycle | `GET /api/journal` — confirm reconciliation clean, review vetoes |
| Post-close | Compare `GET /api/portfolio` equity against the broker directly |
| Weekly | `GET /api/attribution` — is the return from selection or beta? |
| Monthly | Re-snapshot the MCP surface, rotate credentials, restore-from-backup drill |

**Read the veto summary daily**, not just the fills. `GET /api/journal/veto-summary`
is the first place to look when the system seems quiet. Zero fills with hundreds
of vetoes is a broken system that looks identical to a calm market from the
outside.

---

## 2. Incident: reconciliation break

**Severity: critical. This is the worst alert the system emits.**

The ledger and the broker disagree about what is owned. Trading has already
halted automatically. Until resolved, *every* sizing decision is computed against
a book that may not exist.

The usual cause is the `isError` trap: an MCP tool returned `isError: true` with
an HTTP 200, a rejected order was booked as a fill, and the ledger now believes it
holds a position it does not own.

1. **Do not reset the breaker.** It is doing its job.
2. Identify the divergence kind in the journal:
   - `phantom_position` — we think we hold it, the broker disagrees. Most likely
     a rejected order booked as a fill.
   - `unrecorded_position` — the broker holds it, we never booked it. A fill
     arrived after a crash, or the journal write was lost.
   - `quantity_mismatch` — partial fill accounting.
3. Treat **the broker as truth**. Correct the ledger, never the broker, and
   record the correction as a *new* journal record. The journal is append-only; a
   record you can rewrite is not evidence of anything.
4. Find the root cause before resuming. A break that "went away" did not.
5. Reset only after the cause is understood:
   ```bash
   curl -XPOST localhost:8000/api/control/breakers/reset \
        -H 'content-type: application/json' -d '{"acknowledged_by":"your name"}'
   ```

---

## 3. Incident: breaker tripped

Automatic halt on any of: daily loss ≥ 3%, drawdown ≥ 10%, 5 consecutive losing
days, MCP schema drift, or ledger divergence.

**A breaker firing is the system working.** The failure would be it *not* firing.

1. Identify which breaker and why: `GET /api/portfolio` → `breakers`, or the
   `breaker_tripped` journal records.
2. Loss/drawdown breakers: verify the loss is *real* rather than a bad mark. A
   stale or missing price produces a fictitious drawdown.
3. Schema drift: run the drift check. A server-side rename must break the build,
   not a trade.
4. Reset manually, with a name attached. There is no automatic path — a fuse that
   resets itself is not a fuse.

---

## 3b. Incident: cannot connect to Robinhood

1. `python -m osiris.connect --status` — are credentials cached at all?
2. Re-authorize: `python -m osiris.connect`. The consent window expires in about
   five minutes; a timeout looks like a failure.
3. `address already in use` on the callback port means another Osiris process is
   holding it. Close it and retry.
4. If a required capability reports UNAVAILABLE, the tool is absent from *your*
   account's surface rather than broken. Agentic access and the relevant
   permissions must be enabled on the Robinhood side.
5. `Incompatible mcp SDK` means the installed SDK renamed its transport entry
   point. Osiris needs `mcp>=2.0.0`.

---

## 4. Incident: nothing is trading

Symptoms: cycles complete, zero fills.

1. `GET /api/journal/veto-summary`. The top veto code names the cause.
2. Common and legitimate:
   - `insufficient_buying_power` — fully invested. Working as intended.
   - `sector_deviation` / `sector_weight_cap` — the ranker wants concentration the
     kernel will not allow. Expected when momentum crowds into one sector.
   - `position_floor` — refusing to drop below the diversification minimum.
   - `notional_cap` — **expected for new positions.** The per-order cap is 2%
     while symbol exposure is 10%, so positions scale in over several sessions
     *by design*. It is not a bug and not a misconfiguration.
3. `missing_invalidation` means the planner emitted intents with no falsifiable
   exit condition. That is a real defect in the cognition layer, not a risk event.
4. Zero *intents* rather than zero fills is a data-plane or ranking failure. Check
   for `ERROR` journal records at the `ranking` stage.

---

## 5. Incident: agent process is dead

1. The book is unmanaged but positions persist — the broker holds them
   regardless.
2. Engage the kill switch **before** restarting. A supervisor that restarts a
   halted agent into an un-halted state is a real failure mode, and the
   file-based switch is what prevents it.
3. Restart, then let the cycle reconcile *before* it decides anything. It does
   this by construction: reconciliation is step one.
4. Confirm reconciliation clean, then release the switch.

---

## 6. Going live

```bash
.venv/bin/python -m osiris.runner.gate --sessions 120 --json
```

Exit code 0 means cleared. Non-zero means do not commit capital. Blocking checks:

- `account_type_known` — cash vs margin governs settlement and sizing. Trading a
  cash account as margin produces good-faith violations rather than errors: the
  orders fill, and the restriction lands weeks later.
- `explicit_arming` — live mode *and* an explicit risk acknowledgement. Two
  independent affirmations, so no single stray env var arms real money.
- `risk_limits_coherent` — limits must not deadlock against each other.
- `kill_switch_clear` / `kill_switch_writable` — the switch is probed by actually
  writing the file. Discovering the directory is read-only during an incident is
  too late, and the operator would believe they had halted when they had not.
- `breakers_clear`
- `mcp_snapshot_present` — without a committed baseline, drift is undetectable.
- `zero_reconciliation_breaks` — and never reconciling at all is *not* a pass.
- `zero_kernel_bypasses` — placements must never exceed reviews.
- `veto_visibility` — intents with zero fills is a blocking failure.
- `evaluation_gates` — all required gates, and a *missing* gate blocks too. Only
  checking that present gates passed would let a harness that stopped running the
  Monte Carlo test quietly lower the bar.
- `no_fabricated_returns` — a −100% day on a long-only book is an accounting
  artifact from a zero equity mark, and it silently corrupts every metric.

Advisory checks (visible, not blocking): `minimum_viable_capital`,
`paper_duration`, `return_sample_size`, `paper_matches_backtest`.

Then, in order: fund with an amount you would shrug off losing entirely →
smallest tradeable size → `OSIRIS_MODE=live` **and**
`OSIRIS_I_UNDERSTAND_THE_RISK=yes` → configure `OSIRIS_ALERT_WEBHOOK` → watch two
weeks for **zero reconciliation breaks**, not for profit.

If backtest is great and paper is bad, **the backtest is wrong.** Fix it before
proceeding; the cause is almost always lookahead or unmodeled cost.

---

## 7. Alerting

`OSIRIS_ALERT_WEBHOOK` accepts any JSON webhook (Slack, Discord, ntfy, Pushover).

Alert delivery failures are swallowed by design and only logged. A missed
notification is recoverable; an order that failed because its *notification*
failed is not.

Repeats are suppressed per `(kind, title)`. A tripped breaker is re-evaluated
every cycle, so without suppression a single halt would emit one alert per cycle
until a human intervened — and the operator would mute the channel, losing the
critical alerts too.

`critical` is reserved for reconciliation breaks, breaker trips, and kill-switch
events. Fills are `info`: paging on expected behavior trains you to ignore the
channel.

---

## 8. What not to do

- **Do not widen a limit to make an order pass.** The kernel is the reason this
  system is safe to run unattended. If a limit is genuinely wrong, change it
  deliberately, in `.env`, with the reasoning recorded — never in response to a
  specific blocked trade.
- **Do not edit the journal.** Corrections are new records. An append-only log
  you amend is not an audit trail and not tax substantiation.
- **Do not reset a breaker to "see if it happens again."** It will, and you will
  have traded on a broken ledger in the meantime.
- **Do not scale size and universe together.** One variable at a time, or
  attribution stops being interpretable and you cannot tell what worked.
- **Do not trust a passing aggregate.** Read the advisories. Each one is a known
  gap that a green checkmark is hiding.
