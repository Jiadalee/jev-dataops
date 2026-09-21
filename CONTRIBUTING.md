# Contributing

Create a focused branch, install `pip install -e '.[dev]'`, and run `pytest -q` before proposing changes. For model training changes also install `.[train]` and run the local transformer smoke test.

Keep demos synthetic and label model-derived judgments separately from human reference labels. Preserve incomplete-run blocking, group isolation, server-side secrets and bounded data processing. Include an actionable test when changing selection, caching, upload, training or evaluation behavior.

Bug reports should include package version, configuration without credentials, a minimal synthetic input and the expected/actual behavior. Never attach raw customer data, API keys, private cache databases or model weights you cannot redistribute.
