#!/usr/bin/env bash
#
# get_mover.sh — download + verify (+ optionally extract) the MOVER dataset.
#
# Built for Linux. Uses curl (resumable), a temporary chmod-600 .netrc so the
# password never lands in your shell history or the process list, and md5sum
# verification against MOVER's own listing.
#
# USAGE
#   Credentials via env (recommended for HPC / non-interactive):
#     MOVER_USER=max.haberbusch MOVER_PASS='9Gn4Ag%5NqoFmtG6' ./get_mover.sh
#   Or run it and be prompted:
#     ./get_mover.sh
#
# OPTIONS (env vars)
#   MOVER_DEST=./mover         target directory (default ./mover)
#   INCLUDE_WAVEFORMS=1        also fetch the v2 waveform tarballs (large; default off)
#   EXTRACT=1                  untar each *.tar.gz after a successful md5 check
#
# Access expires ~2 weeks after grant. The script is re-runnable: curl resumes
# partial files, so just run it again if a transfer drops. On a login node,
# wrap in tmux or:  nohup ./get_mover.sh > mover.log 2>&1 &
#
set -euo pipefail

# ---------------------------------------------------------------- config
HOST="mover-download.ics.uci.edu"
BASE="https://${HOST}"
DEST="${MOVER_DEST:-./mover}"
INCLUDE_WAVEFORMS="${INCLUDE_WAVEFORMS:-0}"
EXTRACT="${EXTRACT:-0}"

# Small listing files — fetched first so we can preview sizes and verify later.
LISTINGS=( all_md5sum_listing.txt all_size_listing.txt )

# The study-relevant subset (structured/EMR + flowsheets). No waveforms needed
# for MAP forecasting; the covariate lives in the EMR/flowsheet tables.
CORE=(
  README.tar.gz
  sis_emr.tar.gz
  Epic_flowsheets_cleaned.tar.gz
  EPIC_EMR.tar.gz
  EPIC_patient_measurments.tar.gz
)

# Optional raw waveforms — v2 per UCI's instruction. Big. Off by default.
WAVEFORMS=(
  sis_wave_v2.tar.gz
  epic_wave_1_v2.tar.gz
  epic_wave_2_v2.tar.gz
  epic_wave_3_v2.tar.gz
)

# ---------------------------------------------------------------- preflight
for tool in curl md5sum tar awk; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found in PATH." >&2; exit 1; }
done

mkdir -p "$DEST"

: "${MOVER_USER:=}"
if [[ -z "$MOVER_USER" ]]; then read -rp "MOVER username: " MOVER_USER; fi
if [[ -z "${MOVER_PASS:-}" ]]; then read -rsp "MOVER password: " MOVER_PASS; echo; fi

# Temp .netrc keeps the password out of `ps` and history; removed on any exit.
NETRC="$(mktemp)"; chmod 600 "$NETRC"
printf 'machine %s\nlogin %s\npassword %s\n' "$HOST" "$MOVER_USER" "$MOVER_PASS" > "$NETRC"
trap 'rm -f "$NETRC"' EXIT

log(){ printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

fetch(){  # fetch <filename>
  local f="$1"
  log "downloading $f"
  curl --fail --location --continue-at - \
       --retry 10 --retry-delay 10 \
       --netrc-file "$NETRC" \
       -o "$DEST/$f" "$BASE/$f"
}

verify(){  # verify <filename> -> 0 ok / 1 mismatch / 2 not-in-listing
  local f="$1" listing="$DEST/all_md5sum_listing.txt" got want
  got="$(md5sum "$DEST/$f" | awk '{print $1}')"
  # Match by basename in the listing; fall back to "hash present anywhere".
  want="$(awk -v n="$f" 'index($0,n){print $1; exit}' "$listing" || true)"
  if [[ -z "$want" ]]; then
    grep -q "$got" "$listing" && return 0 || return 2
  fi
  [[ "$got" == "$want" ]] && return 0 || return 1
}

# ---------------------------------------------------------------- run
# Build the working set.
SET=( "${LISTINGS[@]}" "${CORE[@]}" )
if [[ "$INCLUDE_WAVEFORMS" == "1" ]]; then SET+=( "${WAVEFORMS[@]}" ); fi

# Grab listings first so size preview + verification are available.
for f in "${LISTINGS[@]}"; do fetch "$f"; done

log "size of selected files (from all_size_listing.txt):"
for f in "${CORE[@]}" $( [[ "$INCLUDE_WAVEFORMS" == "1" ]] && printf '%s ' "${WAVEFORMS[@]}" ); do
  grep -E "(^|[[:space:]/])$f([[:space:]]|$)" "$DEST/all_size_listing.txt" || echo "  (size for $f not listed)"
done
echo

# Download the tarballs.
for f in "${CORE[@]}" $( [[ "$INCLUDE_WAVEFORMS" == "1" ]] && printf '%s ' "${WAVEFORMS[@]}" ); do
  fetch "$f"
done

# Verify every tarball.
fail=0
for f in "$DEST"/*.tar.gz; do
  b="$(basename "$f")"
  if verify "$b"; then
    log "OK   $b"
  else
    case $? in
      1) log "FAIL $b  (md5 mismatch — re-run to resume/repair)"; fail=1 ;;
      2) log "WARN $b  (no md5 in listing; not verified)" ;;
    esac
  fi
done
[[ "$fail" == "1" ]] && { echo "One or more files failed md5. Re-run the script to resume." >&2; exit 1; }

# Optional extraction (only after a clean verify pass).
if [[ "$EXTRACT" == "1" ]]; then
  for f in "$DEST"/*.tar.gz; do
    log "extracting $(basename "$f")"
    tar -xzf "$f" -C "$DEST"
  done
fi

log "done. files in: $DEST"
