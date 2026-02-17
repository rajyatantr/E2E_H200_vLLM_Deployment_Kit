#!/usr/bin/env bash
# =============================================================================
# build_and_push.sh - Build Docker image and push to E2E Container Registry
# =============================================================================
# Usage:
#   ./build_and_push.sh                    # Build and push with defaults
#   ./build_and_push.sh --build-only       # Build without pushing
#   ./build_and_push.sh --tag v1.0         # Custom tag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

BUILD_ONLY=false
TAG="${IMAGE_TAG}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only) BUILD_ONLY=true; shift ;;
        --tag)        TAG="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./build_and_push.sh [--build-only] [--tag TAG]"
            exit 0
            ;;
        *) err "Unknown: $1"; exit 1 ;;
    esac
done

FULL_IMAGE="${E2E_REGISTRY}/${IMAGE_NAME}:${TAG}"

info "Building Docker image: ${FULL_IMAGE}"
docker build -t "${IMAGE_NAME}:${TAG}" -t "${FULL_IMAGE}" "${SCRIPT_DIR}"
ok "Image built: ${IMAGE_NAME}:${TAG}"

if [ "${BUILD_ONLY}" = true ]; then
    info "Skipping push (--build-only)"
    exit 0
fi

info "Logging into E2E Container Registry..."
docker login "${E2E_REGISTRY}"

info "Pushing to ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"
ok "Pushed: ${FULL_IMAGE}"

echo ""
echo "========================================="
echo "  Image: ${FULL_IMAGE}"
echo ""
echo "  To run on an E2E instance:"
echo "    docker pull ${FULL_IMAGE}"
echo "    docker run --gpus all -p 8000:8000 \\"
echo "      -v /home/jovyan/models:/models \\"
echo "      ${FULL_IMAGE}"
echo "========================================="
