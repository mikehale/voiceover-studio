#!/bin/bash
# Builds "Voiceover Studio.app" and a .dmg on an Apple Silicon Mac with Xcode command-line tools.
# Usage: ./build_app.sh            (run from this folder; output in ./dist)
set -euo pipefail
cd "$(dirname "$0")"
SRC="$PWD"; CACHE="$SRC/cache"; DIST="$SRC/dist"; APP="$DIST/Voiceover Studio.app"
UV_VER=0.12.24
FF_BASE=https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2
[ "$(uname -m)" = arm64 ] || { echo "Build on an Apple Silicon Mac"; exit 1; }
mkdir -p "$CACHE" "$DIST"

fetch() { [ -s "$CACHE/$2" ] || /usr/bin/curl -fL --retry 3 -o "$CACHE/$2" "$1"; }
echo "== fetching bundled tools"
fetch "https://github.com/astral-sh/uv/releases/download/$UV_VER/uv-aarch64-apple-darwin.tar.gz" uv.tar.gz
fetch "$FF_BASE/ffmpeg.zip" ffmpeg.zip
fetch "$FF_BASE/ffprobe.zip" ffprobe.zip

echo "== compiling shell"
/bin/rm -rf "$APP" "$DIST/Voiceover Studio.dmg" "$DIST/dmgroot"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin"
xcrun swiftc -O -target arm64-apple-macos12 swift/main.swift -o "$APP/Contents/MacOS/VoiceoverStudio" \
  -framework Cocoa -framework WebKit -framework UniformTypeIdentifiers
cp swift/Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

echo "== bundling tools"
T=$(mktemp -d)
tar -xzf "$CACHE/uv.tar.gz" -C "$T"; cp "$T"/uv-aarch64-apple-darwin/uv "$APP/Contents/Resources/bin/uv"
/usr/bin/ditto -x -k "$CACHE/ffmpeg.zip" "$T/ff"; /usr/bin/ditto -x -k "$CACHE/ffprobe.zip" "$T/ff"
cp "$T/ff/ffmpeg" "$T/ff/ffprobe" "$APP/Contents/Resources/bin/"
/bin/rm -rf "$T"
chmod 755 "$APP/Contents/Resources/bin/"*
xattr -cr "$APP/Contents/Resources/bin" || true

echo "== copying app code (no symlinks, no caches)"
mkdir -p "$APP/Contents/Resources/app"
rsync -a --no-links --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  app/server.py app/requirements.lock app/ui "$APP/Contents/Resources/app/"
rsync -a --no-links --exclude '__pycache__' --exclude '*.pyc' --exclude 'models' pipeline "$APP/Contents/Resources/app/"
if [ -n "$(find "$APP" -type l)" ]; then echo "ERROR: symlinks in bundle"; find "$APP" -type l; exit 1; fi

echo "== icon"
IS="$T.iconset"; /bin/rm -rf "$IS"; mkdir -p "$IS"
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon_1024.png --out "$IS/icon_${s}x${s}.png" >/dev/null
  d=$((s*2)); sips -z $d $d build/icon_1024.png --out "$IS/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$IS" -o "$APP/Contents/Resources/AppIcon.icns"; /bin/rm -rf "$IS"

echo "== ad-hoc signing"
codesign --force -s - "$APP/Contents/Resources/bin/uv"
codesign --force --deep -s - "$APP"
codesign --verify --deep --strict "$APP" && echo "signature ok (ad-hoc)"

echo "== dmg"
R="$DIST/dmgroot"; mkdir -p "$R"
cp -R "$APP" "$R/"; ln -s /Applications "$R/Applications"
cp README.md "$R/Read Me.md"
hdiutil create -volname "Voiceover Studio" -srcfolder "$R" -ov -format UDZO "$DIST/Voiceover Studio.dmg" >/dev/null
/bin/rm -rf "$R"
(cd "$DIST" && /bin/rm -f "Voiceover Studio.zip" && /usr/bin/ditto -c -k --keepParent "Voiceover Studio.app" "Voiceover Studio.zip")
du -sh "$APP" "$DIST/Voiceover Studio.dmg" "$DIST/Voiceover Studio.zip"
echo "done: $DIST"
