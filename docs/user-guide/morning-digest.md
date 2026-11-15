# Morning Digest

A personalized daily briefing that collects data from your connected services, synthesizes a spoken narrative with a local LLM, and delivers it as audio via text-to-speech.

## Quickstart (5 minutes)

### 1. Install and set up OpenJarvis

```bash
git clone https://github.com/open-jarvis/OpenJarvis.git
cd OpenJarvis
uv sync --extra dev
```

### 2. Start a local LLM via Ollama

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3.5:9b    # or any model you prefer
```

### 3. Configure the digest

Edit `~/.openjarvis/config.toml`:

```toml
[engine]
default = "ollama"

[intelligence]
default_model = "qwen3.5:9b"

[digest]
enabled = true
schedule = "0 6 * * *"          # 6 AM daily (cron syntax)
timezone = "America/Los_Angeles"
persona = "jarvis"
honorific = "sir"               # or "ma'am", "boss", etc.
tts_backend = "cartesia"        # or "openai"
voice_id = "c8f7835e-28a3-4f0c-80d7-c1302ac62aae"  # Alistair (British male)
voice_speed = 1.2
sections = ["health", "messages", "calendar", "world"]

[digest.health]
sources = ["oura"]

[digest.messages]
sources = ["gmail", "google_tasks", "slack", "imessage"]

[digest.calendar]
sources = ["gcalendar"]

[digest.world]
sources = ["weather", "hackernews", "news_rss"]
```

### 4. Connect your data sources

```bash
# Google (one flow covers Gmail, Calendar, Tasks, Contacts, Drive)
jarvis connect gdrive
# Paste: <client_id>:<client_secret> — browser opens automatically

# Oura Ring (personal access token)
jarvis connect oura
# Paste your token from https://cloud.ouraring.com/personal-access-tokens

# Spotify
jarvis connect spotify

# Strava
jarvis connect strava
```

For Weather, GitHub, and News — save credential files directly:

```bash
# Weather (OpenWeatherMap — free at https://openweathermap.org/api)
echo '{"api_key": "YOUR_KEY", "location": "San Francisco,CA,US"}' > ~/.openjarvis/connectors/weather.json

# GitHub notifications (token from https://github.com/settings/tokens)
echo '{"token": "ghp_YOUR_TOKEN"}' > ~/.openjarvis/connectors/github.json

# News RSS (no auth needed — configure your feeds)
cat > ~/.openjarvis/connectors/news_rss.json << 'EOF'
{"feeds": [
  {"name": "Arxiv CS.AI", "url": "https://rss.arxiv.org/rss/cs.AI"},
  {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
  {"name": "Bloomberg", "url": "https://feeds.bloomberg.com/markets/news.rss"},
  {"name": "WSJ", "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"}
]}
EOF
```

Hacker News, iMessage, and Apple Music work automatically on macOS with no setup.

### 5. Set your TTS API key

```bash
# Cartesia (sign up at https://play.cartesia.ai)
export CARTESIA_API_KEY="sk_car_..."

# Or OpenAI (https://platform.openai.com/api-keys)
export OPENAI_API_KEY="sk-proj-..."
```

### 6. Run your first digest

```bash
CARTESIA_API_KEY="sk_car_..." jarvis digest --fresh
```

The digest will:
1. Collect data from all connected sources
2. Synthesize a spoken briefing with Qwen3.5 9B
3. Generate audio with the Cartesia Alistair voice
4. Print the text and play the audio

## CLI Commands

```bash
jarvis digest --fresh          # Generate a new digest now
jarvis digest                  # Show today's cached digest
jarvis digest --text-only      # Print text without audio
jarvis digest --history        # Show past digests
jarvis digest --schedule "0 6 * * *"   # Set daily schedule
jarvis digest --schedule off   # Disable schedule
jarvis digest --schedule       # Show current schedule
```

## Saying "Good morning"

When chatting with Jarvis (via CLI, desktop, or browser), saying "Good morning" or "morning digest" automatically triggers the digest — no need to use the `digest` command explicitly.

## Configuration Reference

### Sections

The `sections` list controls what the digest covers, in order of priority:

| Section | Sources | What it provides |
|---------|---------|-----------------|
| `health` | `oura`, `apple_health`, `strava` | Sleep, readiness, activity, workouts |
| `messages` | `gmail`, `google_tasks`, `slack`, `notion`, `imessage`, `github_notifications` | Email triage, tasks, texts, Slack, PRs |
| `calendar` | `gcalendar` | Today's events and schedule |
| `world` | `weather`, `hackernews`, `news_rss` | Weather forecast, tech news, RSS feeds |
| `music` | `spotify`, `apple_music` | Recently played tracks (opt-in) |

### TTS Voices

**Cartesia** (recommended — natural, expressive):
| Voice | ID | Description |
|-------|----|-------------|
| Alistair | `c8f7835e-28a3-4f0c-80d7-c1302ac62aae` | Sophisticated British male |
| Benedict | `3c0f09d6-e0d7-499c-a594-70c5b7b93048` | Polished, formal British male |
| Harrison | `df89f42f-f285-4613-adbf-14eedcec4c9e` | Crisp, professional British male |
| Sterling | `b134c304-d095-4d2b-a77a-914f5e8e84e7` | Deep, commanding, dignified |

**OpenAI TTS**:
| Voice | Description |
|-------|-------------|
| `onyx` | Deep male |
| `nova` | Female, warm |
| `alloy` | Neutral |
| `shimmer` | Female, expressive |

### Persona

The `persona` field loads a prompt file from `configs/openjarvis/prompts/personas/{name}.md`. The default `jarvis` persona delivers briefings with dry British wit, prioritizes urgent items, and interprets health data as trends rather than raw numbers.

To create a custom persona, add a new `.md` file in the personas directory.

### News Feeds

Add any RSS or Atom feed to `~/.openjarvis/connectors/news_rss.json`:

```json
{"feeds": [
  {"name": "Arxiv CS.AI", "url": "https://rss.arxiv.org/rss/cs.AI"},
  {"name": "Arxiv CS.LG", "url": "https://rss.arxiv.org/rss/cs.LG"},
  {"name": "NYT Top Stories", "url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"},
  {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
  {"name": "Bloomberg Markets", "url": "https://feeds.bloomberg.com/markets/news.rss"},
  {"name": "WSJ World News", "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"},
  {"name": "Hacker News", "url": "https://hnrss.org/frontpage"}
]}
```

### Delivered to Telegram, on several schedules

`jarvis serve` runs the digest itself and pushes it to the chat channel when
`[digest] enabled = true` (it also needs a channel: `[channel] enabled`). The
text goes first; the voice note follows, reusing the WAV the digest already
synthesized (Kokoro) instead of synthesizing twice.

```toml
[digest]
enabled = true
schedules = ["0 8 * * 1-5", "0 10 * * 0,6"]  # weekdays 08:00, weekends 10:00
timezone = "Europe/Madrid"                   # crons are evaluated here
language = "Spanish"
honorific = "Carlos"
tts_backend = "kokoro"
voice_id = "ef_dora"
deliver_channel = "telegram"                 # "" = only store the digest
deliver_to = ""                              # "" = first allowed_chat_ids
deliver_voice = true
status_files = ['F:\Backups\Nas\BamboNas\vigilante_estado.txt']
status_max_age_minutes = 30
```

- `schedules` is a list because cron uses commas (`0,6`). The digest tasks in
  `scheduler.db` are reconciled with it at every start: edit and restart.
- A cron without a timezone is evaluated in UTC, so `timezone` matters.
- `status_files`: the first line of each file is appended to the text
  verbatim (never spoken or paraphrased). A file older than
  `status_max_age_minutes`, or missing, is flagged with ⚠ — so a monitor that
  stopped writing shows up in the morning instead of looking healthy.
- If a scheduled run fails, the chat gets "⚠ <label>: <error>" instead of
  silence.

### Weather without an API key: `open_meteo`

The `weather` connector needs an OpenWeatherMap key and reports °F. The
`open_meteo` connector needs nothing but a place name and reports °C:

```python
from openjarvis.connectors.open_meteo import OpenMeteoConnector
OpenMeteoConnector().configure("Vigo")   # writes ~/.openjarvis/connectors/open_meteo.json
```

Then add `"open_meteo"` to `[digest.world] sources`.

## API Endpoints

The digest is also available via the FastAPI server:

```bash
jarvis serve  # Start the server

# GET  /api/digest           — Get today's digest text
# GET  /api/digest/audio     — Stream the digest audio (MP3)
# POST /api/digest/generate  — Force re-generation
# GET  /api/digest/history   — Past digests
# GET  /api/digest/schedule  — Current schedule config
# POST /api/digest/schedule  — Update schedule {"enabled": true, "cron": "0 6 * * *"}
```

## Frontend

The desktop and browser apps show an inline audio player when a digest is generated. The "Connect" buttons in the setup wizard handle OAuth flows automatically — click to connect, authorize in the browser popup, done.

## Troubleshooting

**"No digest for today"** — Run `jarvis digest --fresh` to generate one.

**Empty sections** — Check connector status with `jarvis connect --list`. Ensure tokens haven't expired (Google/Spotify tokens expire after 1 hour and are auto-refreshed on next use).

**Weather not working** — OpenWeatherMap API keys can take up to 2 hours to activate after creation. Use the format `City,State,Country` (e.g., `Palo Alto,CA,US`).

**GitHub 403** — Your personal access token needs the `notifications` permission under Account permissions (not Repository permissions).

**Audio not playing** — Ensure `CARTESIA_API_KEY` or `OPENAI_API_KEY` is set. Check credits at https://play.cartesia.ai or https://platform.openai.com.
