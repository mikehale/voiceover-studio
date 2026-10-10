// Bundled terminal entrypoint. Resolves its own location (including PATH symlinks),
// then replaces itself with the app's private Python. No GUI, shell, or system Python.
import Foundation
import Darwin
import MachO

func fail(_ message: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data(("voiceover-studio: " + message + "\n").utf8))
    exit(code)
}

let fm = FileManager.default
var size: UInt32 = 0
_NSGetExecutablePath(nil, &size)
var buffer = [CChar](repeating: 0, count: Int(size))
guard _NSGetExecutablePath(&buffer, &size) == 0 else { fail("Cannot locate the CLI executable.") }
let executable = URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath()
let contents = executable.deletingLastPathComponent().deletingLastPathComponent()
let resources = contents.appendingPathComponent("Resources")
let codeRoot = resources.appendingPathComponent("app")
let plist = NSDictionary(contentsOf: contents.appendingPathComponent("Info.plist"))
let version = plist?["CFBundleShortVersionString"] as? String ?? "development"
let help = """
Voiceover Studio \(version)
Usage: voiceover-studio [--data-dir PATH] COMMAND [OPTIONS]

Commands:
  process VIDEO ...       Process a local video (process --help for all options)
  benchmark run ...       Measure a pipeline sample with timing and memory reports
  benchmark clone ...     Measure a sustained sample from a saved clone job
  benchmark compare A B   Compare two benchmark result directories
  doctor [--json]         Check app setup and optional cloned voices
  install [--bin-dir DIR] Add a command symlink (default: ~/.local/bin)

Options:
  --help                  Show this help without opening the app
  --version               Show the bundled app version
  --data-dir PATH         Use another app data directory (before COMMAND)

Open Voiceover Studio once to finish model and runtime setup before processing.
"""
var args = Array(CommandLine.arguments.dropFirst())
let originalArgs = args
var data = ProcessInfo.processInfo.environment["VOICEOVER_STUDIO_DATA_DIR"] ??
    fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Voiceover Studio").path
if args.first == "--data-dir" {
    guard args.count >= 2, !args[1].isEmpty else { fail("--data-dir requires a path.", 2) }
    data = args[1]; args.removeFirst(2)
} else if let first = args.first, first.hasPrefix("--data-dir=") {
    data = String(first.dropFirst("--data-dir=".count)); args.removeFirst()
    if data.isEmpty { fail("--data-dir requires a path.", 2) }
}
data = URL(fileURLWithPath: (data as NSString).expandingTildeInPath).standardizedFileURL.path
if args.isEmpty || args == ["--help"] || args == ["-h"] { print(help); exit(0) }
if args == ["--version"] { print("Voiceover Studio \(version)"); exit(0) }

if args.first == "install" {
    let rest = Array(args.dropFirst())
    if rest == ["--help"] || rest == ["-h"] {
        print("Usage: voiceover-studio install [--bin-dir DIR]\nCreates a symlink; never replaces an existing command or edits shell profiles.")
        exit(0)
    }
    var directory = fm.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path
    if !rest.isEmpty {
        guard rest.count == 2 && rest[0] == "--bin-dir" else { fail("Usage: voiceover-studio install [--bin-dir DIR]", 2) }
        directory = (rest[1] as NSString).expandingTildeInPath
    }
    let dir = URL(fileURLWithPath: directory).standardizedFileURL
    let link = dir.appendingPathComponent("voiceover-studio")
    if let target = try? fm.destinationOfSymbolicLink(atPath: link.path) {
        let existing = URL(fileURLWithPath: target, relativeTo: dir).resolvingSymlinksInPath()
        if existing == executable { print("Already installed: \(link.path)"); exit(0) }
        fail("A different symlink already exists at \(link.path). Choose another --bin-dir.")
    }
    if fm.fileExists(atPath: link.path) { fail("A file already exists at \(link.path). Choose another --bin-dir.") }
    do {
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: link.path, withDestinationPath: executable.path)
    } catch { fail("Could not install command: \(error.localizedDescription)") }
    print("Installed: \(link.path)\nAdd \(dir.path) to your shell's PATH if needed. Keep the app at its current location.")
    exit(0)
}

let python = data + "/venv/bin/python"
if !fm.isExecutableFile(atPath: python) {
    if args == ["doctor", "--json"] {
        let report: [String: Any] = ["version": version, "data": data, "ready": false,
                                   "checks": ["python": false], "error": "Open Voiceover Studio to complete setup."]
        let json = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        FileHandle.standardOutput.write(json); print(""); exit(1)
    }
    fail("Private Python is missing at \(python). Open Voiceover Studio to complete setup.", 69)
}
for key in ["PYTHONHOME", "VIRTUAL_ENV", "PYTHONUSERBASE", "PYTHONINSPECT", "PYTHONSTARTUP"] { unsetenv(key) }
for (key, value) in ["PYTHONPATH": codeRoot.appendingPathComponent("pipeline").path,
                     "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                     "PYTHONPYCACHEPREFIX": data + "/pycache", "VOICEOVER_STUDIO_DATA_DIR": data] {
    setenv(key, value, 1)
}
let command = [python, "-s", "-u", "-m", "voiceover_studio"] + originalArgs
var pointers = command.map { strdup($0) } + [nil]
execv(python, &pointers)
fail("Could not start private Python: \(String(cString: strerror(errno)))", 69)
