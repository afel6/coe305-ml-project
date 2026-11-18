"""Build report/report.pdf from report/report.md.

markdown -> HTML (python-markdown, tables extension) -> PDF via headless Microsoft Edge, Google Chrome or Chromium
(no LaTeX or pandoc needed).  Run from the project root:  python report/build_pdf.py
The PDF gets page numbers (from page 2) and a bookmark outline of the headings. The browser prints to
report.partial.pdf, which replaces report.pdf only once it is written, so a failed build keeps the previous PDF.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; @bottom-center { content: counter(page); font-size: 8pt; color: #666; } }
@page :first { @bottom-center { content: none; } }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 9.8pt; line-height: 1.35; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #2a78d6; padding-bottom: 3px; margin-top: 18px; page-break-after: avoid; }
h2 { font-size: 13pt; color: #184f95; margin-top: 14px; page-break-after: avoid; }
h3 { font-size: 11pt; margin-top: 10px; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6px 0 4px 0; font-size: 8.6pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9c8c3; padding: 2px 6px; text-align: right; }
th { background: #eef4fc; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; width: 82%; display: block; margin: 6px auto 2px auto; page-break-inside: avoid; }
.caption { font-size: 8.8pt; color: #444; text-align: center; margin: 0 0 8px 0; }
.cover { text-align: center; padding-top: 70mm; page-break-after: always; }
.cover h1 { border: none; font-size: 24pt; }
.cover table { margin: 18px auto; font-size: 12pt; }
.cover td, .cover th { padding: 6px 18px; }
code { font-size: 9pt; background: #f4f4f2; padding: 0 2px; }
"""

BROWSER_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                 "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Chromium.app/Contents/MacOS/Chromium"]
BROWSER_NAMES = ["msedge", "microsoft-edge", "microsoft-edge-stable", "google-chrome", "google-chrome-stable",
                 "chromium", "chromium-browser", "chrome"]


def browser_candidates(which=shutil.which):
    """Known install paths on Windows and macOS, then browsers on PATH (Linux, or any other install)."""
    return BROWSER_PATHS + [which(name) for name in BROWSER_NAMES]


def print_pdf(candidates, html_uri, pdf, run=subprocess.run, profile=None):
    """Print the page with the first browser that starts; returns its path, or None when none is installed.

    profile: a fresh user-data folder, so the headless browser never hands the job to an open browser window."""
    extra = [f"--user-data-dir={profile}"] if profile else []
    for exe in candidates:
        if not exe:
            continue
        try:
            run([exe, "--headless", "--disable-gpu", *extra, "--no-pdf-header-footer", "--generate-pdf-document-outline",
                 f"--print-to-pdf={pdf}", html_uri], check=True, timeout=180)
            return exe
        except FileNotFoundError:          # not installed at this path: try the next one
            continue
    return None


def to_html(md_text):
    body = markdown.markdown(md_text, extensions=["tables", "attr_list", "md_in_html", "sane_lists", "toc"])
    return f"<!doctype html><html><head><meta charset='utf-8'><title>COE305 report</title><style>{CSS}</style></head><body>{body}</body></html>"


def why(e):
    """The operating system's short reason for a file error (no path, no stack trace)."""
    return e.strerror or type(e).__name__


def fail(message):
    print(message, file=sys.stderr)
    return 1


def discard(part, message):
    """fail(message) after deleting the partial PDF of a failed print; the line says so when it cannot be deleted."""
    try:
        part.unlink(missing_ok=True)
    except OSError as e:
        message += f" {part.name} could not be deleted ({why(e)})."
    return fail(message)


def main(here=HERE, candidates=None, run=subprocess.run):
    try:
        md_text = (here / "report.md").read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return fail("report.md is not UTF-8 text.")
    except OSError as e:
        return fail(f"Could not read report.md: {why(e)}.")
    out_html = here / "report.html"
    try:
        out_html.write_text(to_html(md_text), encoding="utf-8")
    except OSError as e:
        return fail(f"Could not write {out_html.name}: {why(e)}.")
    pdf, part = here / "report.pdf", here / "report.partial.pdf"     # report.pdf is replaced only by a finished print
    try:
        part.unlink(missing_ok=True)                 # a leftover of a killed build must not pass for this build's PDF
    except OSError as e:
        return fail(f"Could not delete the leftover {part.name}: {why(e)}.")
    try:
        profile_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except OSError as e:
        return fail(f"Could not create a temporary folder for the browser profile: {why(e)}.")
    # the browser may still hold its profile files for a moment; a leftover temporary folder is harmless
    with profile_dir as profile:
        try:
            used = print_pdf(browser_candidates() if candidates is None else candidates, out_html.as_uri(), part,
                             run=run, profile=profile)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            return discard(part, f"The browser could not print the PDF ({type(e).__name__}); {out_html.name} was written.")
    if used is None:
        return fail("No Microsoft Edge, Google Chrome or Chromium found; install one to build the PDF "
                    f"({out_html.name} was written).")
    try:
        size = part.stat().st_size
        os.replace(part, pdf)
    except FileNotFoundError:
        return fail(f"The browser ({Path(used).name}) finished without writing {pdf.name}; {out_html.name} was written.")
    except OSError as e:
        return fail(f"Could not replace {pdf.name} ({why(e)}); the new PDF is {part.name}.")
    print("wrote", pdf, size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
