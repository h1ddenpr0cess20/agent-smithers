# Getting Started

This guide gets you from zero to a running Matrix bot using the Responses API with OpenAI or xAI.

## Prerequisites

- Python 3.10+
- A Matrix bot account
- An OpenAI or xAI API key
- Optional: access to remote MCP servers you want to expose

## Install

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install .
```

## Configure

Copy the example env file:

```bash
cp .env.example .env
```

Edit `.env` and set:

- `DEFAULT_MODEL`
- `OPENAI_API_KEY` and/or `XAI_API_KEY`
- `MATRIX_SERVER`
- `MATRIX_USERNAME`
- `MATRIX_PASSWORD`
- `MATRIX_CHANNELS`

Optional:

- Adjust `DEFAULT_MODEL`
- Add `MCP_SERVERS`
- Disable hosted tools you do not want

## Run

Installed command:

```bash
infinigpt-matrix --env-file .env
```

As a module:

```bash
python -m infinigpt --env-file .env
```

## First Checks

- Confirm the bot logs in successfully
- Confirm it joins the configured rooms
- Send `.help`
- Send `.ai hello`
- Check `.model` as an admin to confirm server-fetched models loaded
