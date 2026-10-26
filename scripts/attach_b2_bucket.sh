#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Attach the B2 bucket to AIStore as s3://$B2_BUCKET, enable the S3 features clients rely on,
# and prove AIS can list the bucket with the configured key. Idempotent: safe to re-run after
# `make up`. Uses the AIS REST API directly, so the only dependency is curl.
#
# Required env: B2_BUCKET, B2_ENDPOINT, B2_REGION. Optional: AIS_ENDPOINT.
# The AIS target reads the B2 key from its own environment (deploy/docker-compose.yml).
# B2_ENDPOINT must be reachable from the AIS target container; AIS refuses loopback,
# link-local, and CGNAT bucket endpoints.
set -euo pipefail

: "${B2_BUCKET:?set B2_BUCKET (see .env.example)}"
: "${B2_ENDPOINT:?set B2_ENDPOINT (see .env.example)}"
: "${B2_REGION:?set B2_REGION (see .env.example)}"
AIS_ENDPOINT="${AIS_ENDPOINT:-http://127.0.0.1:51080}"

# Same rule as Config; also keeps the name safe to interpolate into the URL and JSON below.
[[ "${B2_BUCKET}" =~ ^[A-Za-z0-9-]{6,63}$ ]] \
  || { echo "error: B2_BUCKET must be 6-63 letters, digits, or hyphens" >&2; exit 1; }

# Feature bit positions from cmn/feat/feat.go (AIS v5.1, the pinned image); the config API
# takes a numeric bitmask, not names.
S3_REVERSE_PROXY=$((1 << 13))   # proxy relays S3 requests; boto3/s3fs don't follow 307s on PUT
S3_USE_PATH_STYLE=$((1 << 14))  # AIS -> backend addressing: https://endpoint/BUCKET/KEY

ais() { curl -fsS -H 'Content-Type: application/json' "$@"; }

# Container health flips before the cluster finishes starting; wait for primary readiness.
echo "waiting for AIS at ${AIS_ENDPOINT} ..."
for _ in $(seq 1 60); do
  curl -fs -o /dev/null "${AIS_ENDPOINT}/v1/health?prr=true" && break
  sleep 2
done
curl -fsS -o /dev/null "${AIS_ENDPOINT}/v1/health?prr=true" \
  || { echo "error: AIS at ${AIS_ENDPOINT} is not ready" >&2; exit 1; }

# 1. Cluster features first: new buckets inherit bucket-scoped flags at creation time.
#    OR into the current mask so features enabled elsewhere are kept.
current=$(ais "${AIS_ENDPOINT}/v1/cluster?what=cluster_config" | sed -n 's/.*"features":"\([0-9]*\)".*/\1/p')
features=$(( ${current:-0} | S3_REVERSE_PROXY | S3_USE_PATH_STYLE ))
ais -o /dev/null -X PUT "${AIS_ENDPOINT}/v1/cluster" \
  -d "{\"action\":\"set-config\",\"value\":{\"features\":\"${features}\"}}"
echo "cluster features: ${current:-0} -> ${features}"

# 2. Bucket props: point AIS at the B2 endpoint/region for this bucket only.
props="{\"extra\":{\"aws\":{\"endpoint\":\"${B2_ENDPOINT}\",\"cloud_region\":\"${B2_REGION}\"}},\"features\":\"${S3_USE_PATH_STYLE}\"}"
url="${AIS_ENDPOINT}/v1/buckets/${B2_BUCKET}?provider=aws"
body=$(mktemp); trap 'rm -f "${body}"' EXIT
status=$(curl -sS -o "${body}" -w '%{http_code}' -H 'Content-Type: application/json' \
  -X POST "${url}" -d "{\"action\":\"create-bck\",\"value\":${props}}")
case "${status}" in
  200) ;;
  409) ais -o /dev/null -X PATCH "${url}" -d "{\"action\":\"set-bprops\",\"value\":${props}}" ;;
  *)   echo "error: create-bck failed (HTTP ${status}): $(cat "${body}")" >&2; exit 1 ;;
esac

# 3. List one object through AIS so a bad key, endpoint, or region fails here, not mid-run.
status=$(curl -sS -o "${body}" -w '%{http_code}' -H 'Content-Type: application/json' \
  -X GET "${url}" -d '{"action":"list","value":{"props":"name","pagesize":1}}')
[[ "${status}" == 200 ]] || {
  echo "error: AIS cannot list s3://${B2_BUCKET} (HTTP ${status}): $(cat "${body}")" >&2
  echo "check B2_ENDPOINT, B2_REGION, and that the key has listBuckets and listFiles" >&2
  exit 1
}
echo "attached s3://${B2_BUCKET} -> ${B2_ENDPOINT} (region ${B2_REGION}) at ${AIS_ENDPOINT}"
