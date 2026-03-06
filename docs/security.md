# Security

## Secrets

- Keep `.env` out of version control
- Restrict permissions on `.env`, for example:

```bash
chmod 600 .env
```

- Treat `store/` as sensitive Matrix device state

## Network Surface

The bot needs outbound access to:

- Your Matrix homeserver
- OpenAI API endpoints
- Any remote MCP servers you configure

## Tool Risk

- Web search can bring untrusted content into the model context
- Code interpreter can produce files and generated outputs
- MCP servers extend the trust boundary beyond the bot itself

Only enable the tools and MCP endpoints you intend to use.
