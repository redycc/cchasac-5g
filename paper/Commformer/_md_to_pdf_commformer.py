import markdown, subprocess, pathlib, urllib.parse, re

root = pathlib.Path(__file__).parent
md_path = root / "CommFormer_paper_notes_zhtw.md"
html_path = root / "_CommFormer_paper_notes_zhtw.html"
pdf_path = root / "CommFormer_paper_notes_zhtw.pdf"

src = md_path.read_text(encoding="utf-8")

block_store = []
inline_store = []

def _stash_block(m):
    block_store.append(m.group(0))
    return f"@@MATHBLOCK{len(block_store) - 1}@@"

def _stash_inline(m):
    inline_store.append(m.group(0))
    return f"@@MATHINLINE{len(inline_store) - 1}@@"

src = re.sub(r"\$\$[\s\S]+?\$\$", _stash_block, src)
src = re.sub(r"(?<!\$)\$(?!\s)([^\$\n]+?)(?<!\s)\$(?!\$)", _stash_inline, src)

body = markdown.markdown(
    src,
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
)

for i, raw in enumerate(block_store):
    body = body.replace(f"@@MATHBLOCK{i}@@", raw)
for i, raw in enumerate(inline_store):
    body = body.replace(f"@@MATHINLINE{i}@@", raw)

html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>CommFormer Paper Notes</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    tags: 'none'
  }},
  svg: {{ fontCache: 'global' }},
  startup: {{
    typeset: true,
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        document.body.setAttribute('data-mathjax-ready', '1');
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script"></script>
<style>
@page {{ size: A4; margin: 14mm 14mm 16mm 14mm; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", sans-serif;
  font-size: 10.5pt; line-height: 1.6; color: #222; margin: 0;
}}
h1 {{ font-size: 20pt; border-bottom: 2px solid #222; padding-bottom: 6px; margin-top: 0; }}
h2 {{ font-size: 14.5pt; border-bottom: 1px solid #bbb; padding-bottom: 3px; margin-top: 26px; }}
h3 {{ font-size: 12pt; margin-top: 18px; color: #1a1a1a; }}
h4 {{ font-size: 11pt; margin-top: 14px; }}
h5 {{ font-size: 10.5pt; margin-top: 12px; color: #333; }}
p, li {{ margin: 4px 0; }}
code {{
  font-family: Consolas, "Courier New", monospace;
  background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt;
}}
pre {{
  background: #f5f5f5; padding: 10px; border-radius: 4px;
  overflow-x: auto; font-size: 9pt; line-height: 1.4;
  white-space: pre-wrap; word-break: break-word;
}}
pre code {{ background: transparent; padding: 0; }}
table {{
  border-collapse: collapse; width: 100%;
  margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid;
}}
th, td {{ border: 1px solid #bbb; padding: 5px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eaeaea; font-weight: 600; }}
blockquote {{
  border-left: 3px solid #888; margin: 10px 0;
  padding: 6px 14px; color: #444; background: #f8f8f8;
}}
a {{ color: #0366d6; text-decoration: none; word-break: break-all; }}
ul, ol {{ padding-left: 22px; margin: 4px 0; }}
strong {{ color: #111; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 18px 0; }}
h2, h3, h4 {{ page-break-after: avoid; }}
mjx-container {{ font-size: 1em !important; }}
mjx-container[display="true"] {{ margin: 8px 0 !important; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

html_path.write_text(html, encoding="utf-8")

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
url = "file:///" + urllib.parse.quote(str(html_path).replace("\\", "/"))
cmd = [
    edge,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=30000",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}",
    url,
]
print("Running Edge headless (waiting for MathJax)...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
print("stdout:", result.stdout[-500:] if result.stdout else "(empty)")
print("stderr:", result.stderr[-500:] if result.stderr else "(empty)")
print("PDF:", pdf_path, "exists:", pdf_path.exists(), "size:", pdf_path.stat().st_size if pdf_path.exists() else "n/a")
