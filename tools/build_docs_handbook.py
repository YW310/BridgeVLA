"""Bundle all docs Markdown and the algorithm explainer into one offline handbook.

Install tools/requirements-docs.txt once, then run this script from any directory.
Source Markdown remains authoritative; the HTML is a deterministic reading copy.
"""
import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/handbook.html"
TEMPLATE = ROOT / "docs/assets/handbook.fragment.html"
ALGORITHM = ROOT / "docs/assets/algorithm-overview.fragment.html"
BASE_CSS = ROOT / "docs/assets/algorithm-overview-base.css"
GROUPS = (
    ("设计与实现契约", ("docs/design/role-relation-prior.md",)),
    ("数据准备与几何", ("docs/guides/replay.md", "docs/guides/oracle-replay.md",
                    "docs/guides/semantic-gt.md", "docs/oht/data.md")),
    ("安装、配置与训练", ("docs/guides/installation.md", "docs/guides/training.md",
                      "docs/guides/object-conditioning.md", "docs/oht/training.md")),
    ("评估、部署与执行", ("docs/guides/evaluation.md", "docs/oht/runtime.md")),
    ("研究依据与边界", ("docs/research/object-centric-policy-memory.md",)),
    ("历史实验与发布记录", ("docs/experiments/results.md",)),
    ("代码与验证索引", ("docs/reference/code-map.md",)),
    ("附录：入口说明与快速开始", ("README.md", "finetune/OHT/README.md", "docs/README.md")),
)
TOPICS = {
    "docs/design/role-relation-prior.md": ("对象、关系与策略设计", "从研究动机到当前架构，区分已经接线的模块、独立消融与后续 memory / 真机规划。"),
    "docs/guides/replay.md": ("RLBench：从 raw demos 到动作 replay", "先建立可复用的基础动作数据，再考虑物体或角色增强；不要把源数据目录与输出目录混用。"),
    "docs/guides/oracle-replay.md": ("RLBench：Oracle 实例增强", "明确对象点集来自哪里、如何绑定帧与 handle，以及哪些检查只验证身份而不是几何。"),
    "docs/guides/semantic-gt.md": ("RLBench：Semantic-GT 与审计", "沿着 manifest → 角色重写 → 校验 → 训练交接阅读；契约兼容与严格验收分别处理。"),
    "docs/oht/data.md": ("OHT：数据构建、角色标注与几何", "先核对 V4 观测 / 未来标签与时间，再核对相机、深度和工作区，最后做预览与数据验收。"),
    "docs/guides/installation.md": ("环境与依赖", "OHT 数据/模型服务与 RLBench 模拟器依赖不同，先按实验路线选择环境。"),
    "docs/guides/training.md": ("BridgeVLA 基础训练", "预训练、benchmark fine-tuning、数据目录和多卡日志；对象分支的细节在下一节。"),
    "docs/guides/object-conditioning.md": ("O2：配置、训练、评估与排错", "按 object 来源和动作特征路由选配置，训练后再核对闭环统计、对象归因与可视化。"),
    "docs/oht/training.md": ("OHT：连续轨迹模型与训练", "先看已确认 regression_state 基线及证据边界，再看动作表示、状态注入、损失和运行命令。"),
    "docs/guides/evaluation.md": ("Benchmark 闭环评估", "RLBench、COLOSSEUM 和 GemBench 的评估入口；O2 的诊断与统计细节见 O2 专节。"),
    "docs/oht/runtime.md": ("OHT：服务、执行与回放", "按 server → client → 请求协议 → 执行器 → GT 回放 → 验收组织，分清物理时钟与墙钟开销。"),
    "docs/research/object-centric-policy-memory.md": ("相关工作：可以借鉴什么，不能推导什么", "保留原调研版本、监督前提与局限，不将论文方法或历史代码状态视为当前实验结论。"),
    "docs/experiments/results.md": ("历史结果与发布记录", "以下是原始报告和历史记录，不替代当前配置、当前 checkpoint 的闭环验收。"),
    "docs/reference/code-map.md": ("函数、配置与验证入口", "把概念映射回代码，并集中查看当前运行限制和已知证据边界。"),
    "README.md": ("项目概览与快速开始", "原项目概述、网络图、快速运行入口和引用信息，作为全书附录保留。"),
    "finetune/OHT/README.md": ("OHT 快速运行入口", "保留面向 OHT 子项目的最短入口，完整工作流已在正文按数据、训练和执行展开。"),
    "docs/README.md": ("文档维护与导航约定", "说明源文档的组织、命令工作目录以及 HTML 的生成与维护方式。"),
}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
             "meta", "param", "source", "track", "wbr"}


def source_id(path):
    return "doc-" + re.sub(r"[^a-z0-9-]", "-", path.with_suffix("").as_posix().lower())


def discover_sources(root=ROOT):
    discovered = set((root / "docs").rglob("*.md"))
    discovered.update(root / name for name in ("README.md", "finetune/OHT/README.md"))
    missing = [str(path) for path in discovered if not path.is_file()]
    if missing:
        raise ValueError("Missing handbook source(s): " + ", ".join(missing))
    groups, remaining = [], set(discovered)
    for title, names in GROUPS:
        paths = [root / name for name in names if root / name in remaining]
        if paths:
            groups.append((title, paths))
            remaining.difference_update(paths)
    if remaining:
        groups.append(("其他文档", sorted(remaining)))
    return groups


def heading_slug(title):
    title = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", title)
    title = html.unescape(re.sub(r"<[^>]+>", "", title)).lower()
    return "".join(c for c in title if c in "-_ " or unicodedata.category(c)[0] in "LNM").replace(" ", "-")


def anchor_id(document, anchor=""):
    return document + ("--" + unquote(anchor) if anchor else "")


class DocumentRewriter(HTMLParser):
    """Namespace anchors, embed images and retain document links inside the book."""

    def __init__(self, source, sources, root=ROOT):
        super().__init__(convert_charrefs=False)
        self.source, self.sources, self.root = source, sources, root
        self.prefix = sources[source.resolve()]
        self.output, self.used_ids = [], set()

    def resolve_url(self, value, image=False):
        parsed = urlsplit(html.unescape(value))
        if parsed.scheme or parsed.netloc:
            if parsed.scheme.lower() in {"javascript", "vbscript", "data"}:
                raise ValueError("Unsupported source URL: " + value)
            if image:
                raise ValueError("Offline handbook requires local images: " + value)
            return value
        target = (self.source.parent / unquote(parsed.path)).resolve() if parsed.path else self.source.resolve()
        if image:
            if not target.is_relative_to(self.root.resolve()):
                raise ValueError("Image outside repository: " + str(target))
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return "data:" + mime + ";base64," + base64.b64encode(target.read_bytes()).decode("ascii")
        if target in self.sources:
            return "#" + anchor_id(self.sources[target], parsed.fragment)
        if target == (self.root / "docs/design/algorithm-overview.html").resolve():
            return "#" + (unquote(parsed.fragment) or "book-algorithm")
        if target == (self.root / "docs/handbook.html").resolve():
            return "#" + (unquote(parsed.fragment) or "handbook-home")
        relative = Path(os.path.relpath(target, self.root / "docs")).as_posix()
        return relative + ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "iframe", "style", "object", "embed"}:
            raise ValueError("Unsupported active HTML in " + str(self.source) + ": " + tag)
        rewritten = []
        for name, value in attrs:
            if name.startswith("on"):
                raise ValueError("Unsupported event attribute in " + str(self.source))
            if name in {"id", "name"} and value:
                namespaced = value if value.startswith(self.prefix + "--") else anchor_id(self.prefix, value)
                if name == "id" and namespaced in self.used_ids:
                    continue
                self.used_ids.add(namespaced)
                value = namespaced
            elif name == "href" and value is not None:
                value = self.resolve_url(value)
            elif name == "src" and value is not None:
                value = self.resolve_url(value, image=tag == "img")
            rewritten.append(name if value is None else name + '="' + html.escape(value, quote=True) + '"')
        self.output.append("<" + tag + (" " + " ".join(rewritten) if rewritten else "") + ">")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.output.append("</" + tag + ">")

    def handle_data(self, data):
        self.output.append(data)

    def handle_entityref(self, name):
        self.output.append("&" + name + ";")

    def handle_charref(self, name):
        self.output.append("&#" + name + ";")

    def handle_comment(self, data):
        self.output.append("<!--" + data + "-->")


def render_markdown(source, source_ids, root=ROOT):
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    original_fence = parser.renderer.rules["fence"]

    def fence(tokens, index, options, env):
        token = tokens[index]
        if token.info.strip() == "mermaid":
            return ('<details class="original-diagram"><summary>原始 Mermaid 架构定义（完整保留）</summary>'
                    '<p>本手册的八张图解与交互算法先提供可读图；此处保留源图全部连接，便于核对。</p>'
                    '<pre><code class="language-mermaid">' + html.escape(token.content) + '</code></pre></details>\n')
        if token.info.strip() == "latex":
            if r"Z=\mathrm{LN}" in token.content:
                formula = "Z = LN([zᶜ_T, zᶜ_R]) + E_T/R\nA = softmax((W_q LN(Qᶠ)) (W_k Z)ᵀ / √d)"
            elif r"Q'_f=Q_f" in token.content:
                formula = "Qᶠ′ = Qᶠ + W_o [A · diag(1, 1 − pᶜ_NULL) · W_v Z]\nMᶠ = sigmoid(D(Fᶠ, Qᶠ′))"
            else:
                formula = token.content
            return '<pre class="book-formula"><code>' + html.escape(formula) + '</code></pre>\n'
        rendered = original_fence(tokens, index, options, env)
        if len(token.content.splitlines()) > 20:
            language = html.escape(token.info.strip() or "示例")
            return '<details class="long-example"><summary>长示例 · ' + language + '（展开查看完整内容）</summary>' + rendered + '</details>\n'
        return rendered

    parser.renderer.rules["fence"] = fence
    text = source.read_text(encoding="utf-8-sig")
    # The two existing display-math blocks remain fully readable without a math CDN.
    text = re.sub(r"(?ms)^\\\[\s*\n(.*?)^\\\]\s*$", lambda match: "```latex\n" + match[1] + "\n```", text)
    tokens = parser.parse(text)
    # Reading-copy navigation is supplied by the book, not repeated at every source boundary.
    removals = set()
    for index, token in enumerate(tokens[:-2]):
        if token.type != "paragraph_open" or tokens[index + 1].type != "inline":
            continue
        content = tokens[index + 1].content
        links = re.findall(r"\[[^\]]+\]\([^)]+\)", content)
        remainder = re.sub(r"\[[^\]]+\]\([^)]+\)", "", content)
        is_navigation = any(marker in content for marker in ("[文档索引]", "[项目首页]"))
        is_reading_path = content.startswith(("阅读导航：", "阅读路径："))
        if (is_navigation and len(links) >= 2 and not remainder.strip(" ·→\n\t")) or is_reading_path:
            removals.update((index, index + 1, index + 2))
    tokens = [token for index, token in enumerate(tokens) if index not in removals]
    prefix, used, headings, title = source_ids[source.resolve()], set(), [], ""
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1]
        label = re.sub(r"[`*_]", "", inline.content)
        level = int(token.tag[1:])
        if not title:
            title = label
        base, count = heading_slug(inline.content), 0
        slug = base
        while slug in used:
            count += 1
            slug = f"{base}-{count}"
        used.add(slug)
        identity = anchor_id(prefix, slug)
        token.attrSet("id", identity)
        token.tag = "h" + str(min(level + 2, 6))
        tokens[index + 2].tag = token.tag
        if level == 2:
            headings.append((identity, label))
    # Keep all wide-table fields readable on phones instead of squeezing 10+ columns.
    for index, token in enumerate(tokens):
        if token.type != "table_open":
            continue
        end = next(i for i in range(index + 1, len(tokens)) if tokens[i].type == "table_close")
        headers = [tokens[i + 1].content for i in range(index, end) if tokens[i].type == "th_open"]
        if len(headers) > 4:
            token.attrSet("class", "wide-table")
            column = 0
            for item in tokens[index:end]:
                if item.type == "tr_open":
                    column = 0
                elif item.type == "td_open":
                    item.attrSet("data-label", re.sub(r"[`*_]", "", headers[column % len(headers)]))
                    column += 1
    rewriter = DocumentRewriter(source, source_ids, root)
    rewriter.feed(parser.renderer.render(tokens, parser.options, {}))
    rewriter.close()
    return title or source.stem, "".join(rewriter.output), headings


def build():
    groups = discover_sources()
    paths = [path for _, group in groups for path in group]
    identities = {path.resolve(): source_id(path.relative_to(ROOT)) for path in paths}
    documents, contents, manifest = [], [], []
    for number, (group_title, group_paths) in enumerate(groups, 1):
        group_id = "book-part-" + str(number)
        contents.append('<details open><summary><a href="#' + group_id + '">' + html.escape(group_title) + '</a></summary><ul>')
        documents.append('<section class="book-part" id="' + group_id + '"><h2>' + str(number) + " / " + html.escape(group_title)
                         + '</h2><a class="text-small" href="#handbook-home">↑ 全书目录</a>')
        for path in group_paths:
            original_title, body, headings = render_markdown(path, identities)
            identity = identities[path.resolve()]
            relative = path.relative_to(ROOT).as_posix()
            title, introduction = TOPICS.get(relative, (original_title, ""))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append({"source": relative, "id": identity, "sha256": digest})
            contents.append('<li><a href="#' + identity + '">' + html.escape(title) + '</a></li>')
            toc = ''.join('<li><a href="#' + item + '">' + html.escape(label) + '</a></li>' for item, label in headings)
            # Replace the source's first H1 presentation with the book's editorial title,
            # keeping its original anchor so incoming links still work.
            first_heading = re.search(r'<h3\b([^>]*)>.*?</h3>', body, re.DOTALL)
            title_anchor = first_heading.group(1) if first_heading else ""
            if first_heading:
                body = body[:first_heading.start()] + body[first_heading.end():]
            documents.append('<article class="book-document" id="' + identity + '"><header><h3' + title_anchor + '>' + html.escape(title) + '</h3>'
                             '<p class="doc-introduction">' + html.escape(introduction) + '</p></header>'
                             '<nav class="doc-toc" aria-label="' + html.escape(title, quote=True) + '的章节"><ul>' + toc + '</ul></nav>'
                             '<div class="doc-body">' + body + '</div>'
                             '<details class="doc-source"><summary>来源与维护</summary><p class="doc-provenance text-small">'
                             + html.escape(relative) + ' · SHA-256：' + digest[:12] + '</p></details></article>')
        documents.append('</section>')
        contents.append('</ul></details>')
    template = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in {
        "<!--COVERAGE-->": f"收录 {len(paths)} 份 Markdown，{len(groups)} 个主题分部，8 张新增图解，以及完整交互算法。固定浅色；图文离线可读；论文外链需联网，代码/配置链接需保留仓库目录。",
        "<!--CONTENTS-->": "\n".join(contents),
        "<!--ALGORITHM-->": ALGORITHM.read_text(encoding="utf-8"),
        "<!--DOCUMENTS-->": "\n".join(documents),
    }.items():
        if template.count(placeholder) != 1:
            raise ValueError("Missing or duplicate handbook placeholder: " + placeholder)
        template = template.replace(placeholder, value)
    manifest.append({"source": ALGORITHM.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(ALGORITHM.read_bytes()).hexdigest()})
    payload = json.dumps(manifest, ensure_ascii=True).replace("<", "\\u003c")
    return ('<!doctype html>\n<html lang="zh-CN" data-theme="light"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
            '<title>BridgeVLA 完整手册 · 架构、数据、训练与执行</title>\n<style>\n' + BASE_CSS.read_text(encoding="utf-8") + '\n'
            ':root[data-theme="light"] { color-scheme: light; --font-size-base: 16px; --font-weight-normal: 400; '
            '--font-sans: -apple-system,system-ui,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; '
            '--muted-foreground: color-mix(in srgb,var(--foreground) 70%,var(--background)); } '
            'body { padding: 0; margin: 0; }\n</style></head><body>\n' + template
            + '\n<script type="application/json" id="book-source-manifest">' + payload + '</script>'
            '<noscript>所有 Markdown 正文均可直接阅读；节点交互和动态生成的算法逐步说明需要启用 JavaScript。</noscript>'
            '\n</body></html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated handbook differs from its sources")
    args = parser.parse_args()
    try:
        expected = build()
    except ImportError as error:
        parser.exit(1, f"Missing build dependency: {error}. Install: python -m pip install -r tools/requirements-docs.txt\n")
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
            parser.exit(1, "Handbook is stale; run python tools/build_docs_handbook.py\n")
        print("Handbook is up to date.")
        return 0
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(expected)
    print(f"Rendered docs/handbook.html ({len(expected.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
