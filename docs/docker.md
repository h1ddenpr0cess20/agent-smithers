# Docker

This guide covers building and running the InfiniGPT Matrix bot with Docker and Docker Compose. The image includes `libolm` for E2E, runs as a non‑root user, and persists sensitive state under `/data`.

## Prerequisites

- Docker 20.10+
- Optional: Docker Compose v2 (`docker compose`)
- A Matrix account for the bot and a `.env` file (see Configuration)

## Build the Image

Build from the repo root:

```bash
docker build -t infinigpt-matrix:latest .
```

What the image does:

- Installs the package from the repo (`pip install .`)
- Runs `infinigpt-matrix` by default with `--env-file /data/.env --store-path /data/store`

## Run with Docker

1) Prepare configuration and store directories on the host:

```bash
mkdir -p store
cp .env.example .env
```

Edit `.env` and set your OpenAI key, Matrix credentials, rooms, and optional MCP settings.

2) Run the container:

```bash
docker run --rm -it \
  --name infinigpt \
  -v "$(pwd)/.env":/data/.env:ro \
  -v "$(pwd)/store":/data/store \
  -v "$(pwd)/images":/data/images \
  infinigpt-matrix:latest
```

Notes:

- The bot does not expose ports; it connects out to Matrix, OpenAI, and any MCP servers you configure.
- Persist `/data/store` to retain device keys for E2E rooms.

## Run with Docker Compose

An example compose service:

```yaml
services:
  infinigpt:
    image: infinigpt-matrix:latest
    user: "YOUR UID:YOUR GID"
    container_name: infinigpt-matrix
    volumes:
      - ./.env:/data/.env:ro
      - ./store:/data/store
      - ./images:/data/images
    command: ["infinigpt-matrix", "--env-file", "/data/.env", "--store-path", "/data/store", "--log-level", "INFO"]
```

Ensure your `store/` directory is writable by the container user.

## Configuration

- File: mount your `.env` at `/data/.env` (read‑only recommended).
- The file should include `OPENAI_API_KEY`, Matrix settings, and any optional MCP/tool flags.

See [Configuration](configuration.md) for the full schema and validation rules.

## Security Notes

- Treat `store/` as sensitive; back it up securely and do not commit it.
