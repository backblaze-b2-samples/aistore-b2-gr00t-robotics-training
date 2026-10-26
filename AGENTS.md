# AGENTS.md

Fine-tune NVIDIA Isaac GR00T on robot episodes stored in Backblaze B2, staged to node-local disk through an AIStore (AIS) cache, with checkpoints written back to B2.

## Repo map

- `src/aistore_b2_gr00t/`: `config.py` (env), `storage.py` (AIS/B2 clients, prefetch, GetBatch and GET staging, upload, backend counters), `artifacts.py` (`_SUCCESS.json` manifests), `finetune.py` (staging, trainer process group, checkpoint sync).
- `scripts/`: `seed_b2.py`, `verify_artifact.py`.
- `scripts/attach_b2_bucket.sh`: attaches the B2 bucket to AIS over REST and probes access.
- `benchmarks/stage_time.py`: B2 direct vs AIS cold vs AIS warm.
- `tests/`: offline unit tests; `tests/test_integration.py` is the opt-in real-B2 suite.
- `deploy/docker-compose.yml`: single-host AIS cluster bound to 127.0.0.1.

## Commands

```bash
make help                 # every target with a one-line description
make setup                # dev env on any OS; clones Isaac-GR00T + demo data
make setup-gpu            # GPU host: Isaac-GR00T environment
make check                # make lint + make test (what CI runs)
make test-integration     # real B2; needs B2_IT=1 and a dedicated bucket in .env
make up && make seed && make run ARGS="--dry-run"
make bench
make verify PREFIX=<prefix> ARGS="--direct"
```

## Invariants

- Unit tests stay offline: no network, no Docker, no GPU, no `.env`.
- Never print or commit `.env`, and never pass B2 keys to child processes. Only the AIS target and the direct-to-B2 tools read them.
- Integration tests write only under `it/<run>/` and never delete.
- The AIS image digest in `deploy/docker-compose.yml` and the `aistore` SDK pin in `pyproject.toml` change together.
- Write the `complete` `_SUCCESS.json` only after all data is uploaded; consumers trust only a `complete` record and verify every listed size and SHA-256. See [Publication protocol](README.md#publication-protocol).
- Shared with `aistore-b2-cosmos-video-curation`: `artifacts.py`, `scripts/attach_b2_bucket.sh`, and `deploy/docker-compose.yml` (except the project name) are identical; `config.py` and `scripts/verify_artifact.py` share behavior. Change both repos together.
