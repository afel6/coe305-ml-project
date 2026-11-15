"""Build report/report.pdf from report/report.md.

markdown -> HTML (python-markdown, tables extension) -> PDF via headless Microsoft Edge/Chrome
(no LaTeX or pandoc needed).  Run from the project root:  python report/build_pdf.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
md_text = (HERE / "report.md").read_text(encoding="utf-8")
body = markdown.markdown(md_text, extensions=["tables", "attr_list", "md_in_html", "sane_lists", "toc"])

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
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
html = f"<!doctype html><html><head><meta charset='utf-8'><title>COE305 report</title><style>{CSS}</style></head><body>{body}</body></html>"
out_html = HERE / "report.html"
out_html.write_text(html, encoding="utf-8")

candidates = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              shutil.which("msedge") or "", shutil.which("google-chrome") or "", shutil.which("chromium") or ""]
browser = next(c for c in candidates if c and Path(c).exists())
pdf = HERE / "report.pdf"
subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=180)
print("wrote", pdf, pdf.stat().st_size, "bytes")
