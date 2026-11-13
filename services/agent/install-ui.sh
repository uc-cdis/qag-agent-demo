#!/bin/sh
# install-ui.sh <src-assets-dir> <dest-dir> <base-path>
set -eu

src="$1"; dst="$2"; base="${3%/}"

[ -n "$base" ] || { echo "install-ui: base path is empty (arg 3)" >&2; exit 1; }
[ -d "$src" ] || { echo "install-ui: no such source dir: $src" >&2; exit 1; }

rm -rf "$dst"
mkdir -p "$(dirname "$dst")"
cp -r "$src" "$dst"
cd "$dst"

find . -type f \( -name '*.js' -o -name '*.css' -o -name '*.map' \) \
  -exec sed -i "s|/__BASE_PATH__|${base}|g" {} +

if grep -rq '__BASE_PATH__' .; then
  echo "install-ui: unsubstituted sentinel remains in $dst" >&2
  exit 1
fi

for ext in js css; do
  set -- index-*."$ext"
  if [ "$#" -ne 1 ] || [ ! -e "$1" ]; then
    echo "install-ui: expected exactly one index-*.$ext in $dst, found $#" >&2
    exit 1
  fi
  ln -sf "$1" "index.$ext"
done

echo "install-ui: $dst -> base '$base' ($(find . -type f | wc -l) files)"
