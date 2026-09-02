#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
pandoc research/chapter-1-draft.md --from markdown --to typst --template templates/ceu-typst.typ -o /tmp/chapter1.typ
# Compatibility patches for Pandoc 3.7 -> Typst 0.15 math (angle.l and \( escapes are Pandoc writer drift, not styling rules)
sed -i 's/angle\.l/⟨/g; s/angle\.r/⟩/g' /tmp/chapter1.typ
sed -i 's/\\(/(/g; s/\\)/)/g' /tmp/chapter1.typ
sed -i 's/\\\[/[/g; s/\\\]/]/g' /tmp/chapter1.typ
sed -i 's/\\\//\//g' /tmp/chapter1.typ
sed -i 's/zws//g' /tmp/chapter1.typ
sed -i 's/bar\.v\.double  /bar.v.double /g' /tmp/chapter1.typ
sed -i 's/•/parbreak() •/g' /tmp/chapter1.typ
sed -i 's/parbreak() parbreak()/parbreak()/g' /tmp/chapter1.typ
typst compile /tmp/chapter1.typ research/chapter-1-draft.pdf
cp /tmp/chapter1.typ research/chapter-1-draft.typ
echo "Built research/chapter-1-draft.pdf ($(du -h research/chapter-1-draft.pdf | cut -f1))"
