# Media

When xAI media generation returns image payloads, the bot writes temporary artifacts locally and uploads them to Matrix. Grok Imagine video outputs are also downloaded to temporary local artifacts before upload.

## Notes

- Artifacts are written under a local `artifacts/` directory near the Matrix store path.
- Uploaded images and videos are sent to Matrix rooms through the normal Matrix media flow.
- These files are implementation artifacts, not part of the public API.
