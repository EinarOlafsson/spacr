#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
T=$WORK/tool.sh
case "${1:-}" in
 audio-qa) "$T" retained-audio-qa tools/tutorials/verify_staged_audio.py --stage "$STAGE" --lesson "$LESSON" --languages en es fr hi it pt-BR ja zh-CN --retained-narration --retained-from "$ORIGINAL" ;;
 finish) "$T" finish tools/tutorials/finish_staged_visual.py --stage "$STAGE" --lesson "$LESSON" --retain-visual-from "$BASELINE_CANDIDATE" ;;
 matrix) export MATRIX_STAGE=$STAGE; "$T" browser-matrix "$WORK/browser_matrix.py" "$TAG" "$LESSON" ;;
 build) "$T" candidate tools/tutorials/build_appended_candidate.py --stage "$STAGE" --baseline "$BASELINE_CANDIDATE" --lesson "$LESSON" --replace-existing --retain-narration "$LESSON" --retain-visual-timing "$LESSON" --host-web ;;
 verify)
  : "${CANDIDATE:?Actual emitted selective candidate required}"
  "$T" verify-candidate tools/tutorials/verify_release_candidate.py "$CANDIDATE"
  "$T" placeholder tools/tutorials/check_placeholder_mutations.py "$CANDIDATE"
  "$T" checkpoint tools/tutorials/checkpoint_release_candidate.py "$CANDIDATE" --include-web-media ;;
 upload|pages|record)
  : "${CANDIDATE:?Actual verified candidate required}"; : "${HF_NAME:?Root-selected new immutable branch required}"
  case "$1" in
   upload) args=(upload "$CANDIDATE" --branch "$HF_NAME" --tag "$HF_NAME-verified") ;;
   pages) args=(pages "$CANDIDATE" --cache-key "$HF_NAME-verified") ;;
   record) args=(record "$CANDIDATE") ;;
  esac
  "$WORK/sp_tool.sh" "$1" tools/tutorials/publish_release_candidate.py "${args[@]}" ;;
 verify-pages)
  : "${CANDIDATE:?Actual candidate required}"
  "$T" verify-pages tools/tutorials/verify_release_candidate.py "$CANDIDATE" --published "$ROOT/docs/source/_extra/tutorials" ;;
 export)
  : "${CANDIDATE:?Actual recorded candidate required}"; : "${HF_NAME:?Actual immutable branch required}"
  : "${VISUAL_REVIEW:?Actual final selective visual review required}"
  : "${SOURCE_TESTS:?Actual frozen-source test receipt required}"
  : "${SOURCE_TEST_COUNT:?Actual exact test count required}"
  : "${SOURCE_PEER:?Actual final source peer receipt required}"
  tests=(--source-tests "$SOURCE_TESTS")
  if [[ -n "${SOURCE_TESTS_SUPPLEMENT:-}" ]]; then tests+=(--source-tests "$SOURCE_TESTS_SUPPLEMENT"); fi
  if [[ -n "${SOURCE_TESTS_EXTRA:-}" ]]; then tests+=(--source-tests "$SOURCE_TESTS_EXTRA"); fi
  if [[ -n "${SOURCE_TESTS_FINAL:-}" ]]; then tests+=(--source-tests "$SOURCE_TESTS_FINAL"); fi
  peers=(--source-peer "$SOURCE_PEER")
  if [[ -n "${SOURCE_PEER_EXTRA:-}" ]]; then peers+=(--source-peer "$SOURCE_PEER_EXTRA"); fi
  if [[ -n "${SOURCE_PEER_FINAL:-}" ]]; then peers+=(--source-peer "$SOURCE_PEER_FINAL"); fi
  if [[ -n "${SOURCE_PEER_LAST:-}" ]]; then peers+=(--source-peer "$SOURCE_PEER_LAST"); fi
  TOOL_CAP=2G "$T" export "$WORK/export_completed02_checkpoint.py" \
   --stage "$STAGE" --candidate "$CANDIDATE" --source-commit "$SOURCE_COMMIT" --hf-name "$HF_NAME" \
   --output "$WORK/published02-checkpoint-export-20261015" \
   --component-decision "$WORK/root-component-decision.json" --card-acceptance "$APPROVED_CARD_EVIDENCE" \
   --visual-review "$VISUAL_REVIEW" "${tests[@]}" --source-test-count "$SOURCE_TEST_COUNT" "${peers[@]}" ;;
 *) echo 'Phases: audio-qa|finish|matrix|build|verify|upload|pages|verify-pages|record|export' >&2;exit 2 ;;
esac
