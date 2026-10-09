// Voiceover Studio - native shell. Bootstraps a private Python runtime with the bundled `uv`, starts the local
// backend (Resources/app/server.py) and shows its UI in a WKWebView. Apple Silicon only.
import Cocoa
import WebKit
import UniformTypeIdentifiers

let appName = "Voiceover Studio"
let fm = FileManager.default
let res = Bundle.main.resourcePath!
let dataDir = (NSSearchPathForDirectoriesInDomains(.applicationSupportDirectory, .userDomainMask, true).first! as NSString)
    .appendingPathComponent(appName)
let venvPy = dataDir + "/venv/bin/python"
let logsDir = dataDir + "/logs"

func appendLog(_ s: String) {
    let p = logsDir + "/app.log"
    if !fm.fileExists(atPath: p) { fm.createFile(atPath: p, contents: nil) }
    if let h = FileHandle(forWritingAtPath: p) {
        h.seekToEndOfFile(); h.write((ISO8601DateFormatter().string(from: Date()) + " " + s + "\n").data(using: .utf8)!); h.closeFile()
    }
}

func baseEnv() -> [String: String] {
    var env = ProcessInfo.processInfo.environment
    env["UV_PYTHON_INSTALL_DIR"] = dataDir + "/python"
    env["UV_NO_CACHE"] = "1"                         // throw-away temp cache; nothing persists
    env["UV_LINK_MODE"] = "copy"
    env.removeValue(forKey: "UV_CACHE_DIR")
    env["UV_PYTHON_PREFERENCE"] = "only-managed"     // never use Homebrew/system Python
    env["UV_NO_CONFIG"] = "1"
    env["PATH"] = res + "/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = dataDir + "/pycache"   // never write __pycache__ inside the signed .app bundle
    env.removeValue(forKey: "PYTHONPATH"); env.removeValue(forKey: "PYTHONHOME"); env.removeValue(forKey: "VIRTUAL_ENV")
    return env
}

/// Runs a command synchronously, logging its output; returns exit status.
func run(_ exe: String, _ args: [String]) -> Int32 {
    let p = Process(); p.executableURL = URL(fileURLWithPath: exe); p.arguments = args; p.environment = baseEnv()
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
    appendLog("$ \(exe) \(args.joined(separator: " "))")
    do { try p.run() } catch { appendLog("failed to start: \(error)"); return -1 }
    let out = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    appendLog(String(data: out, encoding: .utf8) ?? "")
    return p.terminationStatus
}

let loadingHTML = """
<html><head><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;font:14px -apple-system,sans-serif;background:#f5f5f7;color:#1d1d1f}
@media (prefers-color-scheme:dark){body{background:#1c1c1e;color:#f2f2f7}}
.b{text-align:center;max-width:460px}.s{width:28px;height:28px;border:3px solid #ccc;border-top-color:#5b5bf0;border-radius:50%;animation:r 1s linear infinite;margin:0 auto 16px}
@keyframes r{to{transform:rotate(360deg)}}#m{color:#888;margin-top:6px;font-size:12px}
</style></head><body><div class="b"><div class="s" id="sp"></div><b id="t">Starting Voiceover Studio…</b><div id="m"></div></div></body></html>
"""

class AppDelegate: NSObject, NSApplicationDelegate, WKScriptMessageHandler, WKNavigationDelegate, WKUIDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var serverURL: URL?

    func applicationDidFinishLaunching(_ n: Notification) {
        try? fm.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
        buildMenu()
        let cfg = WKWebViewConfiguration()
        cfg.userContentController.add(self, name: "native")
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self; web.uiDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1000, height: 680),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = appName; window.minSize = NSSize(width: 640, height: 480)
        window.contentView = web; window.center(); window.setFrameAutosaveName("main")
        if let vf = (window.screen ?? NSScreen.main)?.visibleFrame {   // never open larger than the usable screen
            var f = window.frame
            f.size.width = min(f.width, vf.width); f.size.height = min(f.height, vf.height)
            f.origin.x = max(vf.minX, min(f.origin.x, vf.maxX - f.width)); f.origin.y = max(vf.minY, min(f.origin.y, vf.maxY - f.height))
            window.setFrame(f, display: false)
        }
        window.makeKeyAndOrderFront(nil)
        web.loadHTMLString(loadingHTML, baseURL: nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.global(qos: .userInitiated).async { self.bootstrap() }
    }

    func status(_ title: String, _ msg: String = "", error: Bool = false) {
        DispatchQueue.main.async {
            let t = title.replacingOccurrences(of: "'", with: "\\'"), m = msg.replacingOccurrences(of: "'", with: "\\'")
            self.web.evaluateJavaScript("document.getElementById('t').textContent='\(t)';document.getElementById('m').textContent='\(m)';"
                + (error ? "document.getElementById('sp').style.display='none';" : ""), completionHandler: nil)
        }
    }

    func bootstrap() {
        #if !arch(arm64)
        status("Voiceover Studio needs a Mac with Apple Silicon (M1 or newer).", error: true); return
        #endif
        let uv = res + "/bin/uv"
        if !fm.fileExists(atPath: venvPy) {
            Thread.sleep(forTimeInterval: 0.3)
            status("First launch: installing a private Python 3.11…", "This happens once and takes about a minute.")
            if run(uv, ["python", "install", "3.11"]) != 0 {
                status("Could not install Python.", "Check your internet connection, then reopen the app. Log: \(logsDir)/app.log", error: true); return
            }
            if run(uv, ["venv", "--python", "3.11", dataDir + "/venv"]) != 0 {
                status("Could not create the Python environment.", "Log: \(logsDir)/app.log", error: true); return
            }
        }
        status("Starting…")
        startServer()
    }

    func startServer() {
        let p = Process(); p.executableURL = URL(fileURLWithPath: venvPy)
        p.arguments = ["-u", res + "/app/server.py", "--resources", res, "--data", dataDir, "--watch-parent"]
        p.environment = baseEnv()
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        var buf = ""
        let logPath = logsDir + "/server-stdout.log"
        fm.createFile(atPath: logPath, contents: nil)
        let logH = FileHandle(forWritingAtPath: logPath)
        pipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil; return }
            logH?.write(d)
            guard self.serverURL == nil, let s = String(data: d, encoding: .utf8) else { return }
            buf += s
            if let r = buf.range(of: "VS_READY ") {
                let rest = buf[r.upperBound...]
                if let nl = rest.firstIndex(of: "\n"), let u = URL(string: String(rest[..<nl]).trimmingCharacters(in: .whitespaces)) {
                    self.serverURL = u
                    DispatchQueue.main.async { self.web.load(URLRequest(url: u)) }
                }
            }
        }
        p.terminationHandler = { pr in
            appendLog("server exited with status \(pr.terminationStatus)")
            if !self.quitting {
                self.status("The background service stopped unexpectedly.", "Reopen the app. Log: \(logPath)", error: true)
                DispatchQueue.main.async { self.web.loadHTMLString(loadingHTML, baseURL: nil)
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        self.status("The background service stopped unexpectedly.", "Reopen the app. Log: \(logPath)", error: true) } }
            }
        }
        do { try p.run(); server = p } catch {
            status("Could not start the background service.", "\(error)", error: true)
        }
    }

    var quitting = false
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) {
        quitting = true
        if let p = server, p.isRunning {
            p.terminate()                                   // SIGTERM: server saves the queue and stops jobs
            let deadline = Date().addingTimeInterval(4)
            while p.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
            if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        }
    }

    // Keep navigation inside the local server; open other links in the browser.
    func webView(_ w: WKWebView, decidePolicyFor a: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = a.request.url, let h = u.host, h != "127.0.0.1", u.scheme?.hasPrefix("http") == true {
            NSWorkspace.shared.open(u); decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }
    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = m; a.runModal(); completionHandler()
    }

    func reply(_ id: Any?, _ val: Any?) {
        let json: String
        if let v = val, let d = try? JSONSerialization.data(withJSONObject: [v], options: []), let s = String(data: d, encoding: .utf8) {
            json = String(s.dropFirst().dropLast())
        } else { json = "null" }
        web.evaluateJavaScript("window.__nativeReply(\(id ?? 0), \(json))", completionHandler: nil)
    }

    func userContentController(_ uc: WKUserContentController, didReceive msg: WKScriptMessage) {
        guard let b = msg.body as? [String: Any], let cmd = b["cmd"] as? String else { return }
        let id = b["id"]
        switch cmd {
        case "pickFiles":
            let p = NSOpenPanel(); p.allowsMultipleSelection = true; p.canChooseDirectories = false
            p.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie] + ["mkv", "webm"].compactMap { UTType(filenameExtension: $0) }
            p.message = "Choose videos with burned-in English subtitles"
            p.beginSheetModal(for: window) { r in self.reply(id, r == .OK ? p.urls.map { $0.path } : []) }
        case "pickFolder":
            let p = NSOpenPanel(); p.canChooseFiles = false; p.canChooseDirectories = true; p.canCreateDirectories = true
            p.prompt = "Choose"
            p.beginSheetModal(for: window) { r in self.reply(id, r == .OK ? p.urls.first?.path : nil) }
        case "reveal":
            if let path = b["path"] as? String { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
            reply(id, true)
        case "openPath":
            if let path = b["path"] as? String {
                let u = URL(fileURLWithPath: path)
                if (b["text"] as? Bool) == true {
                    NSWorkspace.shared.open([u], withApplicationAt: URL(fileURLWithPath: "/System/Applications/TextEdit.app"),
                                            configuration: NSWorkspace.OpenConfiguration(), completionHandler: nil)
                } else { NSWorkspace.shared.open(u) }
            }
            reply(id, true)
        default: reply(id, nil)
        }
    }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let am = NSMenu()
        am.addItem(withTitle: "About \(appName)", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        am.addItem(.separator())
        am.addItem(withTitle: "Hide \(appName)", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        am.addItem(withTitle: "Quit \(appName)", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = am
        let editItem = NSMenuItem(); main.addItem(editItem)
        let em = NSMenu(title: "Edit")
        em.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = em.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z"); redo.keyEquivalentModifierMask = [.command, .shift]
        em.addItem(.separator())
        em.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        em.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        em.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        em.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = em
        let winItem = NSMenuItem(); main.addItem(winItem)
        let wm = NSMenu(title: "Window")
        wm.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        wm.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        winItem.submenu = wm
        NSApp.mainMenu = main; NSApp.windowsMenu = wm
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
