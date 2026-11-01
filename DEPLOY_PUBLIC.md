# Public demo: the Jetson, behind Cloudflare

Goal: let a few friends try the platform at `https://agents.<your-domain>`, with no port
opened on the home router and nobody getting in without logging in.

```
friend's browser ──HTTPS──> Cloudflare Access (email + one-time code, allow-list)
                                 │
                                 │  Cloudflare Tunnel: cloudflared on the Jetson dials OUT,
                                 ▼  so no inbound port, no public IP needed
                      ┌───────────────── Jetson (Docker) ─────────────────┐
                      │ cloudflared ──> frontend nginx ──/────> React app │
                      │                        └──/api──> backend ──> db  │
                      │                                      └──> Ollama  │
                      └───────────────────────────────────────────────────┘
```

Why this setup:
- **The platform has no authentication** (agreed for the exercise). Cloudflare Access
  adds a login in front of everything, with zero code on our side.
- **No port published**: in `docker-compose.public.yml`, only cloudflared can reach
  nginx. That's also what makes the `Cf-Access-Authenticated-User-Email` header
  trustworthy for per-person rate limits.
- **One origin**: nginx serves the app and forwards `/api`, so no CORS and no
  hardcoded `localhost:8000` in the build.
- **Ollama on the Jetson as the LLM**: OpenRouter's free models allow 50 requests a day
  on an account without $10 of credits (about 15 agent runs a day, for everyone).

---

## 0. Move the domain's DNS to Cloudflare (start first: it can take hours)

1. Create a free account on cloudflare.com, **Add a domain**, choose the **Free** plan.
2. Cloudflare imports your existing DNS records. **Check them before switching**,
   especially MX records if the domain receives email: anything missing stops working.
3. Cloudflare shows two nameservers. At your registrar, replace the domain's
   nameservers with those two. If DNSSEC is on at the registrar, turn it off first.
4. Wait for the "your domain is active on Cloudflare" email.

## 1. Prepare the Jetson

```bash
docker compose version          # must be >= 2.24 (the override uses !reset)
git clone <your repo> agent-platform && cd agent-platform   # or git pull
cp .env.example .env
```

In the Jetson's `.env`:
```
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
CLOUDFLARE_TUNNEL_TOKEN=          # step 2
POSTGRES_PASSWORD=<something other than the default>
```
Ollama must listen beyond localhost (`OLLAMA_HOST=0.0.0.0`, see Day 5) so the backend
container can reach it. The first build is slow on ARM (npm and Python dependencies).

## 2. Create the login (Access) BEFORE the public address

In the Zero Trust dashboard (one.dash.cloudflare.com; on first visit, pick a team name
and the Free plan):

1. **Access → Applications → Add an application → Self-hosted**.
2. Domain: `agents.<your-domain>`. Session duration: 24 hours.
3. Policy "Friends": action **Allow**, include **Emails**: your address and theirs.
4. Login method: **One-time PIN** (a code sent by email; nothing to install).

Doing this first means the address is never reachable without the login, even briefly.

## 3. Create the tunnel

1. **Networks → Tunnels → Create a tunnel → Cloudflared**, name it `agent-platform`.
2. Choose Docker as the environment and copy the token (the long string after
   `--token`) into `CLOUDFLARE_TUNNEL_TOKEN` in the Jetson's `.env`. It's a secret:
   never in Git, never in a chat.
3. Public hostname: subdomain `agents`, your domain, service **HTTP**, URL `frontend:80`.

The dashboard's labels change from time to time; the three ideas stay the same.

## 4. Start and check

```bash
docker compose -f docker-compose.yml -f docker-compose.public.yml up -d --build

docker compose -f docker-compose.yml -f docker-compose.public.yml exec frontend nginx -t
curl -s http://127.0.0.1:5173/api/llm          # {"mode": "api", "default_model": "qwen2.5:7b", ...}
docker compose -f docker-compose.yml -f docker-compose.public.yml logs cloudflared | grep -i registered
```

Then:
- open `https://agents.<your-domain>` in a private window: you must get the Cloudflare
  login page first;
- from another machine: `curl http://<jetson-ip>:8000/health` must **fail** (no port);
- try a streamed run from the browser and watch the steps arrive.

## 5. Before inviting friends

- Tell them: **shared space, test documents only** (everyone sees every knowledge base).
- Limits in place: 5 runs per minute per person, 10 MB uploads, Access allow-list.
- If you use OpenRouter instead of Ollama: a dedicated key with a credit limit.
- **Emergency stop** (cuts public access at once, the app keeps running locally):
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.public.yml stop cloudflared
  ```

## Production-scale notes (interview material)

- Access is a perimeter login; a real multi-user platform needs users and permissions
  in the app itself (who owns which KB and agent), not just at the door.
- Rate limits live in one nginx: with several instances they must move to a shared
  store (Redis) or the API gateway.
- One Jetson is a single point of failure: no backups, no failover. Fine for a demo.
