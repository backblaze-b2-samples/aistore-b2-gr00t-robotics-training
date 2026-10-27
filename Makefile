# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

# .env is sourced by the shell, not parsed by make, so quoted values work. Only targets that
# touch real storage load it; unit tests never do.
LOAD_ENV := set -a; [ ! -f .env ] || . ./.env; set +a;
export

COMPOSE        ?= docker compose -f deploy/docker-compose.yml
UV_RUN         := uv run --locked
GR00T_REF      := n1.7-release
GR00T_COMMIT   := 23ace64f17aa5015259b8609d371eb61a357c776
GR00T_DIR      ?= third_party/Isaac-GR00T
DATASET_PREFIX ?= datasets/cube_to_bowl_5
DEMO_DATA      := $(GR00T_DIR)/demo_data/cube_to_bowl_5
DEMO_DATA_DIR  ?= $(CURDIR)/$(DEMO_DATA)

REQUIRE_LFS := git lfs version >/dev/null 2>&1 || { echo "install git-lfs first (https://git-lfs.com)" >&2; exit 1; }

.PHONY: help setup setup-gpu demo-data up attach down seed run bench verify audit test test-integration check lint fmt

help: ## List targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z_-]+:.*## / { printf "  %-17s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: demo-data ## Dev env (any OS): package, linters, pinned Isaac-GR00T checkout + demo data
	uv sync --locked

setup-gpu: demo-data ## GPU host (Linux x86_64, CUDA 12): Isaac-GR00T's own environment
	@test "$$(uname -s)-$$(uname -m)" = Linux-x86_64 || { echo "GR00T training requires Linux x86_64 + CUDA 12" >&2; exit 1; }
	# uv validates local wheel metadata for every platform in the upstream lockfile.
	git -C $(GR00T_DIR) lfs pull --include "scripts/deployment/dgpu/wheels/*.whl" --exclude ""
	cd $(GR00T_DIR) && uv sync --locked --python 3.10

audit: ## Audit the sample application's locked dependencies (network required)
	uv export --locked --all-groups --no-emit-project --format requirements-txt | uvx --from pip-audit==2.10.1 pip-audit --disable-pip --require-hashes -r /dev/stdin

# CPU setup pulls only the demo dataset; setup-gpu also fetches wheels used by the lockfile.
demo-data: $(GR00T_DIR) ## Fetch GR00T demo_data/cube_to_bowl_5 (git-lfs)
	@$(REQUIRE_LFS)
	@test "$$(git -C $(GR00T_DIR) rev-parse HEAD)" = "$(GR00T_COMMIT)" || { echo "GR00T checkout differs from pinned commit $(GR00T_COMMIT)" >&2; exit 1; }
	git -C $(GR00T_DIR) lfs install --local
	git -C $(GR00T_DIR) lfs pull --include "demo_data/cube_to_bowl_5/**"

$(GR00T_DIR):
	@$(REQUIRE_LFS)
	GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --depth 1 --branch $(GR00T_REF) \
	  https://github.com/NVIDIA/Isaac-GR00T.git $@
	@test "$$(git -C $@ rev-parse HEAD)" = "$(GR00T_COMMIT)" || { echo "upstream tag moved; review the pin before setup" >&2; exit 1; }

up: ## Start AIS on 127.0.0.1 and attach s3://$B2_BUCKET
	$(LOAD_ENV) $(COMPOSE) up -d --wait
	@$(MAKE) --no-print-directory attach

attach: ## Attach s3://$B2_BUCKET to the AIS cluster at $AIS_ENDPOINT (local or existing)
	$(LOAD_ENV) ./scripts/attach_b2_bucket.sh

down: ## Stop AIS (cache volume is kept)
	$(LOAD_ENV) $(COMPOSE) down

seed: ## Upload GR00T demo_data/cube_to_bowl_5 to B2
	$(LOAD_ENV) $(UV_RUN) python scripts/seed_b2.py --local-dir $(DEMO_DATA) --prefix $(DATASET_PREFIX)

run: ## Stage via AIS, fine-tune GR00T, sync checkpoints to B2 (ARGS="--dry-run" etc.)
	$(LOAD_ENV) $(UV_RUN) python -m aistore_b2_gr00t.finetune --dataset-prefix $(DATASET_PREFIX) $(ARGS)

bench: ## Stage-time benchmark: B2 direct vs AIS cold vs AIS warm (ARGS="--stagers 4")
	$(LOAD_ENV) $(UV_RUN) python benchmarks/stage_time.py --prefix $(DATASET_PREFIX) $(ARGS)

verify: ## Verify a published dataset or checkpoint (PREFIX=checkpoints/<run>/checkpoint-N, ARGS="--direct")
	@test -n "$(PREFIX)" || { echo "set PREFIX, e.g. make verify PREFIX=$(DATASET_PREFIX)" >&2; exit 2; }
	$(LOAD_ENV) $(UV_RUN) python scripts/verify_artifact.py --prefix "$(PREFIX)" $(ARGS)

test: ## Unit tests (offline, no GPU)
	$(UV_RUN) pytest

# Real B2 from .env (opt in with B2_IT=1). A fresh AIS cluster runs under a separate compose
# project, ports, and cache volume. Host cache overrides are ignored; `down -v` drops the
# test cache. B2 is untouched except for test objects under it/ (see tests/test_integration.py).
IT_COMPOSE := $(COMPOSE) -p aistore-b2-gr00t-it
test-integration: demo-data ## Real-B2 integration tests on a disposable AIS cluster (ports 51180/51181)
	@$(LOAD_ENV) \
	[ "$${B2_IT:-}" = 1 ] || { echo "set B2_IT=1 in .env to run against real B2 (see README)" >&2; exit 2; }; \
	[ -n "$${B2_KEY_ID:-}" ] && [ -n "$${B2_APP_KEY:-}" ] || { echo "B2_KEY_ID and B2_APP_KEY are required in .env" >&2; exit 2; }; \
	export AIS_ENDPOINT=http://127.0.0.1:51180 AIS_PORT=51180 AIS_TARGET_PORT=51181 DEMO_DATA_DIR=$(DEMO_DATA_DIR); \
	export AIS_CACHE_DIR=target_disk0; \
	trap '$(IT_COMPOSE) down -v >/dev/null 2>&1' EXIT; \
	$(IT_COMPOSE) up -d --wait; \
	$(UV_RUN) pytest -m integration -o addopts="" $(ARGS)

check: lint test ## Lint + unit tests (what CI runs)

lint: ## Ruff lint and format check
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

fmt: ## Apply ruff fixes, then format
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .
