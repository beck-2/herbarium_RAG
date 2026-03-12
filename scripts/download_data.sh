#!/usr/bin/env bash
# download_data.sh — Download NAFlora-mini and all 63 CCH2 DwCA collections.
#
# Usage:
#   bash scripts/download_data.sh
#
# Output:
#   data/raw/naflora1m/h22_miniv1_train.tsv
#   data/raw/naflora1m/h22_miniv1_val.tsv
#   data/raw/symbiota/cch2/<CODE>_DwC-A.zip  (extracted to <CODE>/ subdir)

set -euo pipefail

NAFLORA_DIR="data/raw/naflora1m"
CCH2_DIR="data/raw/symbiota/cch2"

mkdir -p "$NAFLORA_DIR" "$CCH2_DIR"

# ── NAFlora-mini ─────────────────────────────────────────────────────────────

echo "=== NAFlora-mini ==="

for SPLIT in train val; do
    FILE="h22_miniv1_${SPLIT}.tsv"
    DEST="$NAFLORA_DIR/$FILE"
    URL="https://raw.githubusercontent.com/dpl10/NAFlora-1M/main/NAFlora-mini/$FILE"
    if [[ -f "$DEST" ]]; then
        echo "  [skip] $FILE already exists"
    else
        echo "  Downloading $FILE ..."
        curl -L --progress-bar "$URL" -o "$DEST"
        echo "  Done: $DEST ($(wc -l < "$DEST") lines)"
    fi
done

# ── CCH2 DwCA collections ────────────────────────────────────────────────────

echo ""
echo "=== CCH2 DwCA collections (63 total) ==="

CCH2_URLS=(
    https://cch2.org/portal/content/dwca/BFRS_DwC-A.zip
    https://cch2.org/portal/content/dwca/BLMAR_DwC-A.zip
    https://cch2.org/portal/content/dwca/BLMRD_DwC-A.zip
    https://cch2.org/portal/content/dwca/BMR_DwC-A.zip
    https://cch2.org/portal/content/dwca/BORR_DwC-A.zip
    https://cch2.org/portal/content/dwca/CATA-Algae_DwC-A.zip
    https://cch2.org/portal/content/dwca/CATA-Vascular_DwC-A.zip
    https://cch2.org/portal/content/dwca/CDA_DwC-A.zip
    https://cch2.org/portal/content/dwca/CHSC-VascularPlants_DwC-A.zip
    https://cch2.org/portal/content/dwca/CSLA_DwC-A.zip
    https://cch2.org/portal/content/dwca/CSUSB_DwC-A.zip
    https://cch2.org/portal/content/dwca/DAV-VascularPlants_DwC-A.zip
    https://cch2.org/portal/content/dwca/ELH_DwC-A.zip
    https://cch2.org/portal/content/dwca/ENF_DwC-A.zip
    https://cch2.org/portal/content/dwca/FSC_DwC-A.zip
    https://cch2.org/portal/content/dwca/GDRC_DwC-A.zip
    https://cch2.org/portal/content/dwca/GMDRC_DwC-A.zip
    https://cch2.org/portal/content/dwca/HREC_DwC-A.zip
    https://cch2.org/portal/content/dwca/HSC_DwC-A.zip
    https://cch2.org/portal/content/dwca/INF_DwC-A.zip
    https://cch2.org/portal/content/dwca/IRVC_DwC-A.zip
    https://cch2.org/portal/content/dwca/JROH_DwC-A.zip
    https://cch2.org/portal/content/dwca/KNFHC-HappyCamp_DwC-A.zip
    https://cch2.org/portal/content/dwca/KNFSC-Scott-Salmon_DwC-A.zip
    https://cch2.org/portal/content/dwca/KNFY_DwC-A.zip
    https://cch2.org/portal/content/dwca/KRRD_DwC-A.zip
    https://cch2.org/portal/content/dwca/LA_DwC-A.zip
    https://cch2.org/portal/content/dwca/LASCA_DwC-A.zip
    https://cch2.org/portal/content/dwca/LOB_DwC-A.zip
    https://cch2.org/portal/content/dwca/MACF_DwC-A.zip
    https://cch2.org/portal/content/dwca/MBNHM_DwC-A.zip
    https://cch2.org/portal/content/dwca/MCCC_DwC-A.zip
    https://cch2.org/portal/content/dwca/NCC_DwC-A.zip
    https://cch2.org/portal/content/dwca/OBI_DwC-A.zip
    https://cch2.org/portal/content/dwca/PASM_DwC-A.zip
    https://cch2.org/portal/content/dwca/PGM_DwC-A.zip
    https://cch2.org/portal/content/dwca/PINN_DwC-A.zip
    https://cch2.org/portal/content/dwca/PPWD_DwC-A.zip
    https://cch2.org/portal/content/dwca/PUA_DwC-A.zip
    https://cch2.org/portal/content/dwca/RENO-V_DwC-A.zip
    https://cch2.org/portal/content/dwca/RSA-Bryophytes_DwC-A.zip
    https://cch2.org/portal/content/dwca/RSA-Microscope_DwC-A.zip
    https://cch2.org/portal/content/dwca/RSA-VascularPlants_DwC-A.zip
    https://cch2.org/portal/content/dwca/RSA-Wood_DwC-A.zip
    https://cch2.org/portal/content/dwca/SACT_DwC-A.zip
    https://cch2.org/portal/content/dwca/SBBG_DwC-A.zip
    https://cch2.org/portal/content/dwca/SCFS_DwC-A.zip
    https://cch2.org/portal/content/dwca/SD_DwC-A.zip
    https://cch2.org/portal/content/dwca/SDBG_DwC-A.zip
    https://cch2.org/portal/content/dwca/SDM_DwC-A.zip
    https://cch2.org/portal/content/dwca/SDSU_DwC-A.zip
    https://cch2.org/portal/content/dwca/SFSU_DwC-A.zip
    https://cch2.org/portal/content/dwca/SFV_DwC-A.zip
    https://cch2.org/portal/content/dwca/SHTC_DwC-A.zip
    https://cch2.org/portal/content/dwca/SJSU_DwC-A.zip
    https://cch2.org/portal/content/dwca/SPIF_DwC-A.zip
    https://cch2.org/portal/content/dwca/STNF_DwC-A.zip
    https://cch2.org/portal/content/dwca/THRI_DwC-A.zip
    https://cch2.org/portal/content/dwca/UCR_DwC-A.zip
    https://cch2.org/portal/content/dwca/UCSB_DwC-A.zip
    https://cch2.org/portal/content/dwca/UCSC_DwC-A.zip
    https://cch2.org/portal/content/dwca/UCSD_DwC-A.zip
    https://cch2.org/portal/content/dwca/WHIS_DwC-A.zip
    https://cch2.org/portal/content/dwca/WMRC_DwC-A.zip
)

TOTAL=${#CCH2_URLS[@]}
COUNT=0

for URL in "${CCH2_URLS[@]}"; do
    COUNT=$((COUNT + 1))
    ZIPNAME=$(basename "$URL")
    CODE="${ZIPNAME/_DwC-A.zip/}"
    EXTRACT_DIR="$CCH2_DIR/$CODE"
    ZIPPATH="$CCH2_DIR/$ZIPNAME"

    if [[ -d "$EXTRACT_DIR" ]]; then
        echo "  [$COUNT/$TOTAL] [skip] $CODE already extracted"
        continue
    fi

    echo "  [$COUNT/$TOTAL] Downloading $CODE ..."
    if curl -L --progress-bar --fail "$URL" -o "$ZIPPATH" 2>&1; then
        mkdir -p "$EXTRACT_DIR"
        unzip -q "$ZIPPATH" -d "$EXTRACT_DIR"
        rm "$ZIPPATH"
        ROWS=$(wc -l < "$EXTRACT_DIR/occurrences.csv" 2>/dev/null || echo "?")
        echo "        → extracted ($ROWS rows in occurrences.csv)"
    else
        echo "        → FAILED (skipping)"
        rm -f "$ZIPPATH"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Done ==="
echo "NAFlora-mini:"
wc -l "$NAFLORA_DIR"/*.tsv 2>/dev/null | grep -v total || echo "  (not found)"
echo ""
echo "CCH2 collections extracted:"
ls "$CCH2_DIR"/ 2>/dev/null | wc -l
echo ""
echo "Total CCH2 occurrences (approximate):"
find "$CCH2_DIR" -name "occurrences.csv" -exec wc -l {} \; 2>/dev/null \
    | awk '{s+=$1} END {print s " lines across all collections"}'
