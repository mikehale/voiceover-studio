# Voiceover Studio

Voiceover Studio turns videos with **burned-in English subtitles** (for example fan-subbed anime on YouTube) into an English voice-over version. It works in five steps:

1. It reads the subtitles from the picture with OCR.
2. It separates the Japanese voices from the music and effects with Demucs.
3. It speaks the subtitles with Kokoro voices. Yellow subtitles get the dialogue voice and cyan subtitles get the narrator voice.
4. It mixes the English voice over the music and effects. By default the Japanese voices are left out of the English track, except the singing in opening/ending songs, which stays at full level; you can change both in **Settings > Mix**. The original Japanese audio is always kept as a second audio track.
5. It writes an .mp4 to `~/Movies/Voiceover`.

**Requirements: a Mac with Apple Silicon (M1 or newer) running macOS 12 or later. Intel Macs are not supported.**

## Install

1. Open `Voiceover Studio.dmg` and drag **Voiceover Studio** onto **Applications**.
2. The app is not signed by an Apple-registered developer, so macOS blocks the first launch. To open it anyway, use one of these:
   - **macOS 15 (Sequoia) or newer:** double-click the app and click **Done** in the warning. Then go to **System Settings > Privacy & Security**, scroll down to the message about Voiceover Studio, click **Open Anyway** and confirm with your password.
   - **macOS 13–14:** right-click (or Control-click) the app, choose **Open**, then click **Open** again.
   - **Terminal (any version):** `xattr -dr com.apple.quarantine "/Applications/Voiceover Studio.app"`

   You only need to do this once.

## First launch

On first launch the app sets itself up. You don't need Homebrew, Python or Terminal for this. A progress screen shows each step:

| Step | Size | Notes |
|---|---|---|
| Private Python 3.11 runtime | ~20 MB | Installed by the bundled `uv` |
| Python packages (PyTorch, Demucs, Kokoro, RapidOCR, OpenCV) | ~900 MB on disk | Pinned versions; no download cache is kept |
| yt-dlp + deno (YouTube downloader and its JavaScript runtime) | ~130 MB | yt-dlp is kept updatable |
| Kokoro voice model (fp16) + voices | 205 MB | |
| Demucs htdemucs model | ~170 MB | |

On a fast connection setup takes about 3–6 minutes. Everything goes into `~/Library/Application Support/Voiceover Studio` (about 1.4 GB). Later launches take a few seconds. To uninstall, delete the app and that folder.

FFmpeg/ffprobe (static arm64 builds) and `uv` are bundled inside the app.

The setup screen also has an unticked **Cloned voices (Chatterbox)** checkbox. Leave it off and nothing extra is installed (see *Cloned voices* below).

## Using it

- **Add videos.** Paste one or more YouTube video or playlist URLs and click **Add to queue**. A playlist is expanded into one queue item per video. **Add local videos…** adds files from your Mac.
- **Process only part of a video.** Fill in *Only process start/end* before adding, for example `4:20` to `6:20`.
- **Watch progress.** The queue runs one item at a time: download, OCR, voice separation, speech, mix, then writing the video. Each item shows its stage and percentage.
- **Manage the queue.** You can reorder (⤒ moves an item to the top of the waiting items, right after the video being processed, without interrupting it; ↑ ↓ move it one place), stop, retry, remove and view the log (≡) for each item. **Pause queue** stops new items from starting.
- **Pick up after quitting.** If you quit mid-video, the queue is saved. On the next launch it continues from the last finished stage of that video.
- **"No hardsubs found".** An item is marked this way when OCR finds almost no subtitle lines in it. Soft subtitles that you can switch on or off in the player don't count.
- **Where the output goes.** Each finished video is saved as `<title> (English VO).mp4` in the output folder. Its OCR subtitle files go in the `Subtitles/` subfolder. If **keep source** is on, the downloaded original goes in `Sources/`.

### Cloned voices (optional)

Cloned voices speak the English lines **in the voices of the original Japanese actors** (so with their accent), using [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT licence) locally on your Mac.

- **Install:** tick the box on the first-run screen, or later use **Settings > Voices > Cloned voices > Install**. It goes into its own Python environment (`clone-venv`) plus the model (`models/chatterbox`) inside the app's data folder: about 4 GB (0.8 GB of Python packages + 3.2 GB model; about 2 minutes on a fast connection) in total. The standard install is not changed. **Remove** deletes both.
- **Use:** tick **Use cloned voices** before adding videos, or switch it per queued item. **Settings > Voices** has a default for new videos (off).
- **How it works:** speakers are grouped automatically by voice fingerprint from the separated Japanese voice track. The narrator (cyan) gets its own voice. Each speaker gets a 6–10 s reference clip of their clean speech. Lines are fitted into the same subtitle slots as the standard voices. A line that can't be cloned uses the standard voice for its colour, and the item's log says so.
- **Speed:** generation runs on the GPU (MPS) with CPU fallback. It is much slower than the standard voices: on an M3 Max, Chatterbox needs about 1.8 s of GPU time per second of speech, so a 4-minute sample took 4.3 minutes instead of 1.3, and a 24-minute episode takes roughly 20–25 minutes longer.
- **German accent (optional):** **Settings > Voices > Accent of cloned voices** sets the default (Original or German), and each queued video has its own **German accent / Original accent** button. German uses Chatterbox Multilingual (language `de`) reading the English text in the same cloned voices. It needs a one-time extra download of about 2.1 GB into the same model folder; this starts the first time you choose German, or use **Download now**. **Remove** deletes it with the rest. Each line falls back to the original-accent clone if the German model can't do it (very short lines such as "And..."), then to the standard voice; the log notes every fallback. It is slower: on an M3 Max about 2.5 s of GPU time per second of speech once running, and the 4-minute sample took 8.9 minutes (original accent: 4.7 minutes) including one retry and one fallback. The accent is subtle and varies by line, and the speech is somewhat less clear than with the original accent.
- **Limits:** grouping by voice is automatic and can mix up similar-sounding characters or split one character in two. Short lines are assigned from the conversation context. Chatterbox output carries Resemble AI's inaudible watermark.

### Settings

- **Output folder.**
- **Voices.** Choose the voice for each subtitle colour: dialogue (yellow), narrator (cyan), and song lyrics (white; not voiced by default). Install or remove cloned voices and set whether new videos use them.
- **Levels.** Japanese voice level (default **Off**: the slider all the way left leaves the Japanese voices out of the English track, so there's no ducking either; older versions used −15 dB, and installs still on that default switch to Off when updating), extra ducking under English (−9 dB, only used when the Japanese voices are on), **Keep song vocals** (default on: during songs with white lyric subtitles the Japanese vocals play at full level with no ducking, whatever the Japanese voice level; songs without lyric subtitles follow the Japanese voice level), and English level.
- **Output video.**
  - **H.264** (default) re-encodes with the Mac's hardware encoder and plays everywhere.
  - **Copy** is faster, but YouTube's AV1 video needs IINA/VLC or an M3-or-newer Mac to play.
- **Download quality.** Up to 720p by default.
- **Browser cookies.** Only for age- or region-restricted videos.
  - Chrome shows a one-time Keychain prompt.
  - Safari needs **System Settings > Privacy & Security > Full Disk Access** for Voiceover Studio.
- **Keep or delete the downloaded source.**
- **Edit corrections…** opens find/replace rules for common OCR mistakes.
- **Update yt-dlp.** Use this when YouTube downloads start failing.

## Speed

On an M-series Mac a 24-minute episode takes very roughly 10–20 minutes. OCR and speech are CPU-bound. Demucs uses the GPU (MPS). H.264 output adds a hardware encode pass.

## Known limitations

- Apple Silicon only.
- The app works only with subtitles that are burned into the picture. Standard voices are picked by subtitle colour (yellow/red/green = dialogue, cyan = narrator, white = lyrics), not per character. Cloned voices group speakers automatically within each video, without character names, and don't remember characters between episodes.
- OCR can misread stylised fonts. Use **Edit corrections…** for recurring mistakes.
- Very long speech in a short subtitle slot is sped up to 1.4× at most, and may overlap the next line.
- YouTube changes often. If downloads fail, use **Settings > Update yt-dlp**.
- Process only videos you have the right to use.

## For the developer: building, signing and notarizing

`build_app.sh` (run on an Apple Silicon Mac with Xcode command-line tools) does the following:

1. Downloads the pinned `uv` and static FFmpeg.
2. Compiles the Swift shell.
3. Assembles the bundle and its icon.
4. Signs the app *ad hoc*.
5. Creates `dist/Voiceover Studio.dmg` and `.zip`.

To distribute without the Gatekeeper warning you would need:

1. An Apple Developer Program membership ($99/yr) and a **Developer ID Application** certificate.
2. To sign every Mach-O inside the bundle, innermost first, with the hardened runtime and a timestamp: `uv`, `ffmpeg`, `ffprobe`, then the app itself. The command is `codesign --force --options runtime --timestamp -s "Developer ID Application: …" …`. Avoid `--deep` for the final signature.
3. Entitlements are probably not needed, because the Python runtime is downloaded later and not inside the bundle. If the app is later changed to bundle Python, it would need `com.apple.security.cs.allow-unsigned-executable-memory` and `disable-library-validation`.
4. To submit with `xcrun notarytool submit "Voiceover Studio.dmg" --apple-id … --team-id … --wait`, sign the dmg too, then run `xcrun stapler staple "Voiceover Studio.dmg"`.
