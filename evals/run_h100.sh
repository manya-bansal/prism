#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
RUNPOD_API="https://api.runpod.io/graphql"
GPU_TYPE="NVIDIA H100 80GB HBM3"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DISK_GB=20
BOOT_TIMEOUT=300   # 5 min
SSH_TIMEOUT=60     # 1 min
POD_MAX_LIFE=600   # 10 min hard kill
DEFAULT_SIZES="2048 4096 6144 8192 12288 16384"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATED_DIR="${REPO_ROOT}/generated"
BENCH_DIR="${SCRIPT_DIR}/h100_bench"
RESULTS_DIR="${SCRIPT_DIR}/results"
SIZES="${1:-$DEFAULT_SIZES}"

POD_ID=""

# --- Prereqs ---
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "ERROR: RUNPOD_API_KEY not set" >&2
    exit 1
fi

for cmd in curl python3 ssh tar; do
    command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not found" >&2; exit 1; }
done

[[ -d "${GENERATED_DIR}" ]] || { echo "ERROR: ${GENERATED_DIR} missing" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"

# --- Sync generated kernel ---
echo "Syncing kernel from generated/..."
cp "${GENERATED_DIR}/cutlass_my_h100_match_no_tail.cu" "${BENCH_DIR}/cutlass_my_h100_match_no_tail.cu"
echo "Kernel sync done."

for f in main.cu utils.cuh cutlass_my_h100_match_no_tail.cu Makefile; do
    [[ -f "${BENCH_DIR}/$f" ]] || { echo "ERROR: ${BENCH_DIR}/$f missing" >&2; exit 1; }
done

# --- JSON helper (replaces jq) ---
json_val() {
    python3 -c "
import json, sys
data = json.load(sys.stdin)
keys = '$1'.split('.')
for k in keys:
    if data is None: break
    data = data.get(k) if isinstance(data, dict) else None
print('' if data is None else data)
"
}

json_ssh() {
    python3 -c "
import json, sys
data = json.load(sys.stdin)
ports = (data.get('data') or {}).get('pod', {})
ports = ((ports or {}).get('runtime') or {}).get('ports') or []
for p in ports:
    if p.get('privatePort') == 22:
        ip, port = p.get('ip', ''), p.get('publicPort', '')
        if ip and port:
            print(f'{ip} {port}')
            break
"
}

# --- Helpers ---
gql() {
    local query="$1"
    local escaped
    escaped=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$query")
    curl -sf -X POST "$RUNPOD_API?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\": $escaped}" \
    || { echo "ERROR: API call failed" >&2; return 1; }
}

# --- Cleanup trap ---
cleanup() {
    if [[ -n "$POD_ID" ]]; then
        echo "Terminating pod ${POD_ID}..."
        gql "mutation { podTerminate(input: { podId: \"${POD_ID}\" }) }" >/dev/null 2>&1 || true
        echo "Pod terminated."
    fi
}
trap cleanup EXIT

# --- Create pod ---
# --- Create pod + wait for SSH (with retries) ---
MAX_POD_ATTEMPTS=30
SSH_CMD=""

for ATTEMPT in $(seq 1 $MAX_POD_ATTEMPTS); do
    echo "Creating H100 pod (attempt ${ATTEMPT}/${MAX_POD_ATTEMPTS})..."
    CREATE_RESP=$(gql "mutation {
        podFindAndDeployOnDemand(input: {
            name: \"h100-eval-$(date +%s)\",
            gpuTypeId: \"${GPU_TYPE}\",
            imageName: \"${IMAGE}\",
            gpuCount: 1,
            volumeInGb: 0,
            containerDiskInGb: ${DISK_GB},
            startSsh: true,
            ports: \"22/tcp\",
            minMemoryInGb: 0,
            minVcpuCount: 0
        }) {
            id
        }
    }")

    POD_ID=$(echo "$CREATE_RESP" | json_val "data.podFindAndDeployOnDemand.id")
    if [[ -z "$POD_ID" ]]; then
        echo "  Failed to create pod. Retrying in 10s..." >&2
        POD_ID=""
        sleep 10
        continue
    fi
    echo "Pod created: ${POD_ID}"

    # Wait for pod to boot & get SSH info
    echo "Waiting for pod to start (timeout: ${BOOT_TIMEOUT}s)..."
    ELAPSED=0
    while (( ELAPSED < BOOT_TIMEOUT )); do
        POD_INFO=$(gql "{ pod(input: { podId: \"${POD_ID}\" }) { id runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } desiredStatus } }")

        UPTIME=$(echo "$POD_INFO" | json_val "data.pod.runtime.uptimeInSeconds")
        STATUS=$(echo "$POD_INFO" | json_val "data.pod.desiredStatus")

        if [[ -n "$UPTIME" ]]; then
            SSH_INFO=$(echo "$POD_INFO" | json_ssh)
            if [[ -n "$SSH_INFO" ]]; then
                SSH_HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
                SSH_PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
                SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p ${SSH_PORT} root@${SSH_HOST}"
                echo "Pod running (uptime: ${UPTIME}s). SSH: root@${SSH_HOST}:${SSH_PORT}"
                break
            fi
            echo "  Pod running but SSH port not available yet..."
        fi

        if [[ "$STATUS" == "EXITED" ]]; then
            echo "  Pod exited unexpectedly." >&2
            break
        fi

        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    # Wait for SSH
    if [[ -n "$SSH_CMD" ]]; then
        echo "Waiting for SSH (timeout: ${SSH_TIMEOUT}s)..."
        SSH_ELAPSED=0
        while (( SSH_ELAPSED < SSH_TIMEOUT )); do
            if $SSH_CMD "echo ok" >/dev/null 2>&1; then
                echo "SSH connected."
                break 2  # break out of both loops
            fi
            sleep 3
            SSH_ELAPSED=$((SSH_ELAPSED + 3))
        done
        echo "  SSH failed. Terminating pod and retrying..."
    else
        echo "  Pod did not provide SSH info. Retrying..."
    fi

    # This attempt failed — kill the pod and try again
    gql "mutation { podTerminate(input: { podId: \"${POD_ID}\" }) }" >/dev/null 2>&1 || true
    POD_ID=""
    SSH_CMD=""
    sleep 5
done

if [[ -z "$SSH_CMD" ]]; then
    echo "ERROR: Failed to get a working pod after ${MAX_POD_ATTEMPTS} attempts" >&2
    exit 1
fi

# --- Remote watchdog ---
echo "Setting ${POD_MAX_LIFE}s remote watchdog..."
$SSH_CMD "nohup bash -c 'sleep ${POD_MAX_LIFE} && kill 1' >/dev/null 2>&1 &"

# --- Upload benchmark ---
echo "Uploading benchmark source..."
$SSH_CMD "mkdir -p /root/bench"
tar -cf - -C "${BENCH_DIR}" main.cu utils.cuh cutlass_my_h100_match_no_tail.cu Makefile \
    | $SSH_CMD "tar -xf - -C /root/bench"
echo "Upload complete."

# --- Compile ---
echo "Compiling..."
$SSH_CMD "export PATH=/usr/local/cuda/bin:\$PATH && cd /root/bench && make test" 2>&1
echo "Compilation done."

# --- Run sweep ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/h100_${TIMESTAMP}.txt"

{
    echo "# H100 Matmul Benchmark Results"
    echo "# Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "# Pod ID: ${POD_ID}"
    echo "# GPU: ${GPU_TYPE}"
    echo "# Image: ${IMAGE}"
    echo "# Sizes: ${SIZES}"
    echo "#"
    $SSH_CMD "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" 2>/dev/null | sed 's/^/# GPU Info: /'
    echo ""
} > "$RESULTS_FILE"

echo "Running benchmark sweep: ${SIZES}"
for SIZE in $SIZES; do
    echo "  Size ${SIZE}..."
    $SSH_CMD "export PATH=/usr/local/cuda/bin:\$PATH && cd /root/bench && stdbuf -oL ./out/test ${SIZE}" 2>&1 | tee -a "$RESULTS_FILE"
done

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo "Done."
