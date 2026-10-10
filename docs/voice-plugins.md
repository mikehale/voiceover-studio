# Voice plugins (format version 1)

A voice plugin adds one cloned voice to Voiceover Studio: a short reference recording of one speaker plus a
manifest. Plugins are installed locally from a `.zip`; the app never downloads voices and ships none. Plugins are
not signed: the SHA-256 hashes in the manifest catch damaged or partial copies, not deliberate changes, so only
install plugins from people you trust.

## Package

A flat `.zip` (no folders), conventionally named `<id>.vsvoice.zip`:

| Entry | Content |
|---|---|
| `manifest.json` | the manifest below (UTF-8 JSON) |
| `<reference>.wav` | the reference recording; any other files must be listed in the manifest too |

Nothing else may be in the zip. Limits: 20 MB per file, 40 MB in total; file names `[A-Za-z0-9][A-Za-z0-9._-]*`, at most 64 characters.

### manifest.json

```json
{
 "format_version": 1,
 "id": "example_narrator",
 "name": "Example Narrator",
 "version": "1.0.0",
 "author": "Jane Doe",
 "accent": "original",
 "sample_rate": 24000,
 "reference": "example_narrator.wav",
 "files": {"example_narrator.wav": "<sha256 hex of the file>"}
}
```

| Field | Rule |
|---|---|
| `format_version` | `1` |
| `id` | 1-40 characters of `a-z 0-9 _`, starting with a letter or digit; also the install folder name. Installing a plugin with the same id replaces the old one. |
| `name`, `version`, `author` | non-empty strings (80 characters max); `name` is shown in the pickers |
| `accent` | `original` (as the reference speaker sounds) or `german_v3` (German accent, Chatterbox Multilingual V3) |
| `sample_rate` | sample rate of the reference WAV in Hz; must match the file |
| `reference` | a `.wav` listed in `files`: PCM, 3-60 s; 10-20 s of one speaker without music works best (mono 24 kHz recommended) |
| `files` | every file in the package except `manifest.json`, with its lower-case hex SHA-256 |

## Checks (on install and on every scan)

A package is rejected, with the reason shown in the app and written to `logs/server.log`, when:

- it is not a zip, has folders, path separators, duplicate or unlisted entries, or files over the limits;
- `manifest.json` is missing or not valid JSON;
- a file's SHA-256 differs from the manifest (damaged or changed after packaging) or a listed file is missing;
- a manifest field is missing or invalid, or the reference is not a readable PCM WAV of 3-60 s at `sample_rate`.

## Installing, listing, removing

- **Settings > Voices > Install voice plugin…** (file picker for `.zip`), or `POST /api/plugins/install {"path": "/path/to/plugin.zip"}`.
  The zip is checked, extracted to `plugins/<id>.installing/`, checked again and then moved to `plugins/<id>/`.
- The app scans `plugins/` at launch and after every install or removal; folders that no longer pass the checks are ignored and listed as errors under Voice plugins.
- **Remove** next to a plugin in Settings, or `POST /api/plugins/remove {"id": "<id>"}`. Settings and waiting videos that used it go back to the voices cloned from the video.
- `GET /api/state` lists installed plugins in `custom_voices` (id, name, accent, version, author) and rejected folders in `plugin_errors`.

## Using a plugin voice

Plugin voices need cloned voices (Chatterbox) installed; a `german_v3` plugin also needs the German-accent model,
which downloads on first use. They appear in **Settings > Voices > Cloned narrator voice / Cloned dialogue voice**
and in the add row's **Voices** picker (narrator, dialogue or all lines). Every line of that role is spoken from the
plugin's reference; fallback per line: plugin voice with its accent -> same voice without the accent -> standard
(Kokoro) voice. A hash of the reference and the accent are part of the cache key of each generated line.

## Making a plugin

```sh
tools/make_voice_plugin.py build --wav example_narrator.wav --id example_narrator --name "Example Narrator" \
    --accent original --version 1.0.0 --author "Jane Doe" --out example_narrator.vsvoice.zip
tools/make_voice_plugin.py verify example_narrator.vsvoice.zip
```

Only package recordings you have the right to use.
