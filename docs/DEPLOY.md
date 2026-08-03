# Running Osiris 24/7

The agent is one always-on process. Deployment is therefore one container on
one small Linux VM. It sleeps until the market schedule wakes it, monitors
held positions every minute while the market is open, and serves the dashboard
the whole time.

**Why not Vercel/Lambda:** serverless wakes on requests and dies in seconds.
Osiris wakes *itself* and holds a stateful broker session. Wrong shape.

**Why not just the laptop:** a closed lid at 9:40 AM means no trading and no
stop-loss enforcement. The machine must be awake every market minute.

## What you need

- A $5/month VM: Hetzner CX22, DigitalOcean basic droplet, or EC2 t3a.small.
  Ubuntu 24.04. The workload is tiny — the box idles 23 hours a day.
- Docker: `curl -fsSL https://get.docker.com | sh`
- This repo on the box: `git clone <your-repo> && cd osiris`

## First-time setup

**1. Copy configuration.** From your Mac:

```bash
scp .env you@server:osiris/.env
```

**2. Copy Robinhood credentials.** OAuth needs a browser, which the server
does not have. Authorize on your Mac once (`python -m osiris.connect`), then
copy the token cache into the container's volume:

```bash
ssh you@server 'docker volume create osiris_osiris-data'
scp -r ~/.osiris you@server:/tmp/osiris-tokens
ssh you@server 'docker run --rm -v osiris_osiris-data:/data -v /tmp/osiris-tokens:/src alpine \
  sh -c "mkdir -p /data/.osiris && cp /src/mcp-auth.json /data/.osiris/ && chmod 600 /data/.osiris/mcp-auth.json" \
  && rm -rf /tmp/osiris-tokens'
```

Tokens refresh themselves from then on.

**3. Start it.**

```bash
docker compose up -d --build
docker compose logs -f     # look for "LIVE — REAL ORDERS" and the schedule
```

`restart: unless-stopped` + Docker's systemd unit means it survives crashes
and reboots. There is nothing to babysit.

## Reaching the dashboard

The container binds to localhost only — the dashboard shows your balance and
exposes the kill switch, so it must not face the internet. Two good options:

```bash
# Option A: SSH tunnel when you want to look.
ssh -N -L 8030:localhost:8030 you@server
# then open http://localhost:8030

# Option B: Tailscale (persistent, phone-friendly).
# Install tailscale on the server and your devices, then:
tailscale serve --bg 8030
# dashboard at https://<server-name>.<tailnet>.ts.net from any of your devices
```

## Operations

| Task | Command (on the server) |
|---|---|
| Watch logs | `docker compose logs -f` |
| Emergency stop | `docker compose exec osiris touch /data/KILL_SWITCH` |
| Resume | via the dashboard, or delete the file |
| Update code | `git pull && docker compose up -d --build` |
| Back up the journal | `docker cp osiris:/data/journal-live.jsonl .` |
| Health check | `curl localhost:8030/api/health` |

The kill switch, journal, and credentials all live on the `osiris-data`
volume: replacing the container loses nothing, and a kill switch engaged
before an update is still engaged after it.

## The safety analysis you should read once

- The container can place real orders. Anyone with root on the VM can trade
  your account. Use SSH keys, disable password auth, keep the box patched.
- `OSIRIS_ALERT_WEBHOOK` in `.env` is worth setting before going remote: a
  Discord/Slack/ntfy webhook that pages you on breaker trips and failures,
  so you hear about problems without watching logs.
- The journal is your audit trail and tax record. Back it up occasionally.
