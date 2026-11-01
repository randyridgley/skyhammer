"""Generate a Material for MkDocs site from the SDK source, examples/ and the project docs.

Run through ``scripts/docs.sh``; this only writes Markdown and ``mkdocs.yml`` into OUT_DIR:

- ``index.md``: README plus feature cards.
- ``guides/``: one page per example (docstring, cost, code, CLI, roadmap features, API used),
  grouped like ``examples/README.md``.
- ``cli.md``: every ``skyhammer`` command, from the Typer app.
- ``reference/``: one mkdocstrings page per module, with the examples that use it.
- ``roadmap.md``, ``design.md``.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

import skyhammer
from skyhammer.cli.main import app

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "skyhammer"
EXAMPLES = ROOT / "examples"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
REPO_URL = PYPROJECT["project"]["urls"]["Repository"]
BLOB_URL = f"{REPO_URL}/blob/main"
SITE_URL = PYPROJECT["project"]["urls"]["Documentation"]

# Modules documented first under "API reference"; the rest go under "Internals".
SKIP_MODULES = {"__init__"}
WORDS = {"api": "API", "openai": "OpenAI", "dpo": "DPO", "nova": "Nova", "sft": "SFT"}
# Docstring markers that become their own sections on a guide page.
MARKER = re.compile(r"\b(Cost|Requires|Run(?: after [^:\n]+)?|CLI(?: equivalents)?):\s*")


# --------------------------------------------------------------------------- roadmap


@dataclass
class Feature:
    id: str
    capability: str
    status: str
    notes: str


def parse_roadmap() -> dict[str, Feature]:
    features = {}
    for line in (ROOT / "ROADMAP.md").read_text().splitlines():
        m = re.match(r"^\| ([A-Z]{2}-\d+) \|(.*)\|$", line)
        if not m:
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", m.group(2))]  # `\|` is a literal pipe
        # | ID | Capability | skyhammer | Milestone | Notes |
        if len(cells) >= 4:
            features[m.group(1)] = Feature(m.group(1), cells[0], cells[1], cells[3])
    return features


# --------------------------------------------------------------------------- examples


@dataclass
class Example:
    path: Path
    section: str
    summary: str  # "What it shows" column
    covers: list[str]
    title: str = ""
    prose: str = ""
    meta: dict[str, str] = field(default_factory=dict)  # Cost/Run/CLI/Requires
    code: str = ""
    symbols: list[str] = field(default_factory=list)  # e.g. "LLM", "LLM.create_sft_job"

    @property
    def slug(self) -> str:
        return re.sub(r"^\d+_", "", self.path.stem).replace("_", "-")

    @property
    def nav_title(self) -> str:
        if self.path.suffix != ".py":
            return self.path.name
        stem = re.sub(r"^\d+_", "", self.path.stem).replace("fine_tune", "fine-tune")
        words = stem.split("_")
        words = [WORDS.get(w, w) for w in words]
        return " ".join([words[0][:1].upper() + words[0][1:], *words[1:]])


def parse_index() -> list[Example]:
    examples, section = [], ""
    for line in (EXAMPLES / "README.md").read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        m = re.match(r"^\| \[([^\]]+)\]\(([^)]+)\) \| (.*) \| (.*) \|$", line)
        if m:
            covers = re.findall(r"[A-Z]{2}-\d+", m.group(4))
            examples.append(Example(EXAMPLES / m.group(2), section, m.group(3), covers))
    return examples


def split_docstring(doc: str) -> tuple[str, str, dict[str, str]]:
    """Title (first paragraph), remaining prose, and Cost/Run/CLI/Requires values."""
    title, _, body = doc.partition("\n\n")
    title = " ".join(title.split())
    prose, meta = [], {}
    parts = MARKER.split(body)
    prose.append(parts[0])
    for key, value in zip(parts[1::2], parts[2::2], strict=True):
        if key.startswith("Run"):
            # A run command ends at the line break; anything after it is prose again.
            value, _, rest = value.partition("\n")
            prose.append(rest)
        meta[key] = value.strip()
    text = " ".join(" ".join(prose).split())
    return title, text, meta


def public_members() -> dict[str, tuple[str, object]]:
    """Name in ``skyhammer.__all__`` -> (defining module, object)."""
    out = {}
    for name in skyhammer.__all__:
        obj = getattr(skyhammer, name)
        module = getattr(obj, "__module__", None)
        if not isinstance(module, str) or not module.startswith("skyhammer."):
            module = _defining_module(name)
        out[name] = (module, obj)
    return out


def _defining_module(name: str) -> str:
    for path in sorted(SRC.glob("*.py")):
        if re.search(rf"^{re.escape(name)}\b\s*[:=]", path.read_text(), re.M):
            return f"skyhammer.{path.stem}"
    return "skyhammer"


def load_example(ex: Example, public: dict[str, tuple[str, object]]) -> None:
    text = ex.path.read_text()
    if ex.path.suffix != ".py":
        ex.title, ex.prose, ex.code = ex.summary, "", text
        ex.meta["Run"] = f"uv run skyhammer apply examples/{ex.path.name} --plan"
        return
    tree = ast.parse(text)
    ex.title, ex.prose, ex.meta = split_docstring(ast.get_docstring(tree) or ex.path.stem)
    # Show the code without the module docstring.
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        text = "\n".join(text.splitlines()[first.end_lineno or 0 :])
    ex.code = text.strip("\n") + "\n"

    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("skyhammer")
        for alias in node.names
        if alias.name in public
    ]
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    ex.symbols = list(dict.fromkeys(imported))
    # Methods called on classes the example imports (e.g. LLM.create_sft_job).
    for name in imported:
        obj = public[name][1]
        if inspect.isclass(obj):
            for attr in sorted(attrs):
                if not attr.startswith("_") and attr in vars(obj):
                    ex.symbols.append(f"{name}.{attr}")


# --------------------------------------------------------------------------- links


@dataclass
class Site:
    out: Path
    pages: dict[str, str]  # repo-relative source path -> site page path
    written: set[Path] = field(default_factory=set)

    def rewrite(self, md: str, source: str, page: str) -> str:
        """Point relative links in ``source`` (repo path) at site pages or GitHub."""
        src_dir = os.path.dirname(source)
        page_dir = os.path.dirname(page)

        def fix(m: re.Match[str]) -> str:
            target = m.group(2)
            if re.match(r"^([a-z]+:|#|/)", target):
                return m.group(0)
            path, _, anchor = target.partition("#")
            repo_path = os.path.normpath(os.path.join(src_dir, path))
            if repo_path in self.pages:
                new = os.path.relpath(self.pages[repo_path], page_dir or ".")
            else:
                new = f"{BLOB_URL}/{repo_path}"
            return f"{m.group(1)}({new}{'#' + anchor if anchor else ''})"

        return re.sub(r"(\[[^\]]*\])\(([^)\s]+)\)", fix, md)

    def write(self, page: str, text: str) -> None:
        self.written.add(self.out / page)
        _write_if_changed(self.out / page, text.rstrip() + "\n")

    def prune(self) -> None:
        """Delete pages left over from removed examples or modules."""
        for path in self.out.rglob("*"):
            if path.is_file() and path not in self.written:
                path.unlink()


def _write_if_changed(dest: Path, text: str) -> None:
    # Unchanged files keep their mtime, so `mkdocs serve` doesn't reload in a loop.
    if dest.exists() and dest.read_text() == text:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)


def ref_page(module: str) -> str:
    return f"reference/{module.removeprefix('skyhammer.').replace('.', '/')}.md"


def symbol_link(symbol: str, public: dict[str, tuple[str, object]], page: str) -> str:
    head = symbol.split(".")[0]
    module = public[head][0]
    rel = os.path.relpath(ref_page(module), os.path.dirname(page))
    return f"[`{symbol}`]({rel}#{module}.{symbol})"


def summary_of(symbol: str, public: dict[str, tuple[str, object]]) -> str:
    obj: object = public[symbol.split(".")[0]][1]
    for part in symbol.split(".")[1:]:
        obj = getattr(obj, part, None)
    doc = inspect.getdoc(obj) if obj is not None and not isinstance(obj, (str, dict)) else None
    return " ".join(doc.split("\n\n")[0].split()) if doc else ""


# --------------------------------------------------------------------------- pages


def guide_page(
    ex: Example, features: dict[str, Feature], public: dict[str, tuple[str, object]]
) -> str:
    page = f"guides/{ex.slug}.md"
    lang = "python" if ex.path.suffix == ".py" else "yaml"
    out = [f"---\ntitle: {ex.nav_title}\n---\n", f"# {ex.title}\n"]
    if ex.covers:
        out.append(" ".join(f"`{fid}`" for fid in ex.covers) + "\n")
    if ex.prose:
        out.append(ex.prose + "\n")
    if "Requires" in ex.meta:
        out.append(f'!!! note "Requires"\n    {ex.meta["Requires"]}\n')
    if "Cost" in ex.meta:
        cost = " ".join(ex.meta["Cost"].split())
        out.append(f'!!! tip "Cost and time"\n    {cost[:1].upper() + cost[1:]}\n')

    out.append(f'## Code\n\n```{lang} title="examples/{ex.path.name}" linenums="1"\n{ex.code}```\n')
    run_key = next((k for k in ex.meta if k.startswith("Run")), None)
    if run_key:
        label = "" if run_key == "Run" else f"{run_key}:\n\n"
        out.append(f"## Run it\n\n{label}```bash\n{ex.meta[run_key]}\n```\n")
    cli_key = next((k for k in ex.meta if k.startswith("CLI")), None)
    if cli_key:
        cmds = [" ".join(c.split()) for c in ex.meta[cli_key].splitlines() if c.strip()]
        out.append("## From the CLI\n\n```bash\n" + "\n".join(cmds) + "\n```\n")

    if ex.covers:
        out.append("## Features\n\n| ID | Capability | Status | Notes |\n|---|---|---|---|")
        for fid in ex.covers:
            f = features.get(fid)
            if f:
                out.append(f"| `{fid}` | {f.capability} | {f.status} | {f.notes} |")
        out.append(
            f"\nSee the [roadmap]({os.path.relpath('roadmap.md', 'guides')}) for every feature.\n"
        )
    if ex.symbols:
        out.append("## API used\n")
        for sym in ex.symbols:
            desc = summary_of(sym, public)
            out.append(f"- {symbol_link(sym, public, page)}" + (f": {desc}" if desc else ""))
        out.append("")
    out.append(f"[View on GitHub]({BLOB_URL}/examples/{ex.path.name})")
    return "\n".join(out)


def reference_page(module: str, examples: list[Example], public: dict[str, tuple]) -> str:
    page = ref_page(module)
    used = [ex for ex in examples if any(public[s.split(".")[0]][0] == module for s in ex.symbols)]
    out = [f"# `{module}`\n"]
    if used:
        links = ", ".join(
            f"[{ex.nav_title}]({os.path.relpath(f'guides/{ex.slug}.md', os.path.dirname(page))})"
            for ex in used
        )
        out.append(f'!!! example "Used in"\n    {links}\n')
    out.append(f"::: {module}\n")
    return "\n".join(out)


def reference_index(public: dict[str, tuple[str, object]]) -> str:
    out = [
        "# API reference\n",
        "Everything below is importable from the top-level package: "
        "`from skyhammer import LLM, Dataset`.\n",
        "| Name | Kind | Summary |",
        "|---|---|---|",
    ]
    for name in sorted(public, key=lambda n: (public[n][0], n)):
        obj = public[name][1]
        kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "constant"
        if inspect.isclass(obj) and issubclass(obj, BaseException):
            kind = "exception"
        out.append(
            f"| {symbol_link(name, public, 'reference/index.md')} | {kind} | "
            f"{summary_of(name, public).replace('|', '/')} |"
        )
    return "\n".join(out)


def cli_page(examples: list[Example]) -> str:
    root = typer.main.get_command(app)
    out = [
        "# CLI reference\n",
        "Every SDK feature is also available from the `skyhammer` command "
        "(`uv run skyhammer --help`).\n",
    ]

    # Typer >=0.27 vendors click, so duck-type groups and options instead of isinstance.
    def walk(cmd: Any, path: list[str], parent: typer.Context | None) -> None:
        ctx = typer.Context(cmd, info_name=path[-1], parent=parent)
        is_group = hasattr(cmd, "list_commands")
        if path != ["skyhammer"] and not cmd.hidden:
            name = " ".join(path)
            level = "#" * min(len(path), 4)
            out.append(f"{level} `{name}`\n")
            if cmd.help:
                out.append(inspect.cleandoc(cmd.help).split("\f")[0] + "\n")
            if not is_group:
                usage = re.sub(r"\{(\w+)\}", lambda m: m.group(1).upper(), cmd.get_usage(ctx))
                out.append(f"```text\n{usage}\n```\n")
                rows = [_param_row(p, ctx) for p in cmd.params if p.name != "help"]
                if rows:
                    out.extend(
                        [
                            "| Argument / option | Type | Default | Description |",
                            "|---|---|---|---|",
                            *rows,
                            "",
                        ]
                    )
            mention = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
            guides = [ex for ex in examples if mention.search(_cli_text(ex))]
            if guides:
                out.append(
                    "Examples: "
                    + ", ".join(f"[{ex.nav_title}](guides/{ex.slug}.md)" for ex in guides)
                    + "\n"
                )
        if is_group:
            for sub_name in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, sub_name)
                if sub:
                    walk(sub, [*path, sub_name], ctx)

    walk(root, ["skyhammer"], None)
    return "\n".join(out)


def _cli_text(ex: Example) -> str:
    return " ".join(" ".join(v.split()) for k, v in ex.meta.items() if k.startswith(("CLI", "Run")))


def _param_row(p: Any, ctx: typer.Context) -> str:
    if p.param_type_name == "option":
        names = ", ".join(f"`{o}`" for o in [*p.opts, *p.secondary_opts])
        help_text = p.help or ""
        default = "" if p.is_flag and not p.default else p.get_default(ctx)
    else:
        names, help_text, default = f"`{(p.name or '').upper()}`", "", ""
        if p.required:
            help_text = "Required."
    if default is None or default == "":
        default_s = ""
    else:
        default_s = f"`{getattr(default, 'value', default)}`"
    choices = getattr(p.type, "choices", None)
    type_s = " \\| ".join(str(c) for c in choices) if choices else p.type.name
    return f"| {names} | {type_s} | {default_s} | {help_text.replace('|', '/')} |"


def home_page(site: Site, sections: dict[str, list[Example]]) -> str:
    readme = (ROOT / "README.md").read_text()
    body = readme.split("\n", 1)[1]  # drop the H1; the hero replaces it
    cards = ['<div class="grid cards" markdown>\n']
    for section, exs in sections.items():
        first = exs[0]
        items = ", ".join(f"[{e.nav_title}](guides/{e.slug}.md)" for e in exs[:6])
        cards.append(f"-   **[{section}](guides/{first.slug}.md)**\n\n    ---\n\n    {items}\n")
    cards.append(
        "-   **[CLI reference](cli.md)**\n\n    ---\n\n"
        "    `skyhammer bootstrap`, `chat`, `apply`, `jobs`, `eval`, `deployments`\n"
    )
    cards.append(
        "-   **[API reference](reference/index.md)**\n\n    ---\n\n"
        "    `LLM`, `Dataset`, `FineTuningJob`, `EvalJob`, `BatchJob`, graders, errors\n"
    )
    cards.append("</div>\n")
    hero = f"# Skyhammer\n\n*{PYPROJECT['project']['description']}*\n\n" + "\n".join(cards)
    return hero + "\n" + site.rewrite(body, "README.md", "index.md")


def mkdocs_yml(
    out: Path, sections: dict[str, list[Example]], modules: list[str], internals: list[str]
) -> str:
    def ref_nav(mods: list[str]) -> str:
        return "".join(f"          - {m.removeprefix('skyhammer.')}: {ref_page(m)}\n" for m in mods)

    guides = "".join(
        f"      - {section}:\n"
        + "".join(f'          - "{e.nav_title}": guides/{e.slug}.md\n' for e in exs)
        for section, exs in sections.items()
    )
    return f"""\
site_name: Skyhammer
site_description: {PYPROJECT["project"]["description"]}
site_url: {SITE_URL}
repo_url: {REPO_URL}
repo_name: skyhammer
edit_uri: ""
docs_dir: {out}
theme:
  name: material
  icon: {{repo: fontawesome/brands/github, logo: material/hammer}}
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: custom
      accent: custom
      toggle: {{icon: material/weather-night, name: Dark mode}}
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: custom
      accent: custom
      toggle: {{icon: material/weather-sunny, name: Light mode}}
  font: {{text: Inter, code: JetBrains Mono}}
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - navigation.footer
    - navigation.instant
    - toc.follow
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate
extra_css: [assets/extra.css]
# `scripts/docs.sh serve` regenerates the pages when any of these change.
watch: [{ROOT / "src"}, {ROOT / "examples"}, {ROOT / "README.md"}, {ROOT / "ROADMAP.md"},
        {ROOT / "DESIGN.md"}, {ROOT / "scripts" / "gen_docs.py"}]
hooks: [{ROOT / "scripts" / "docs_hook.py"}]
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - toc: {{permalink: true}}
  - pymdownx.details
  - pymdownx.inlinehilite
  - pymdownx.superfences
  - pymdownx.highlight: {{anchor_linenums: true, line_spans: __span}}
  - pymdownx.tabbed: {{alternate_style: true}}
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [{ROOT / "src"}]
          options:
            docstring_style: google
            show_root_heading: false
            show_root_toc_entry: false
            show_source: false
            show_if_no_docstring: true
            show_signature_annotations: true
            separate_signature: true
            signature_crossrefs: true
            merge_init_into_class: true
            members_order: source
            filters: ["!^_"]
            heading_level: 2
            show_symbol_type_heading: true
            show_symbol_type_toc: true
nav:
  - Home: index.md
  - Guides:
      - guides/index.md
{guides}  - CLI: cli.md
  - API reference:
      - reference/index.md
      - Public modules:
{ref_nav(modules)}      - Internals:
{ref_nav(internals)}  - Roadmap: roadmap.md
  - Design: design.md
"""


EXTRA_CSS = """\
/* Forge palette: steel-navy sky, ember-orange hammer strike. */
:root, [data-md-color-scheme="default"] {
  --sh-steel: #16202e;
  --sh-ember: #f26b1d;
  --md-primary-fg-color: var(--sh-steel);
  --md-primary-fg-color--light: #2a3a52;
  --md-primary-fg-color--dark: #0b1119;
  --md-primary-bg-color: #f4f6f9;
  --md-accent-fg-color: var(--sh-ember);
  --md-accent-fg-color--transparent: #f26b1d1a;
  --md-typeset-a-color: #c2410c; /* darker ember: 5:1 contrast on white */
}
[data-md-color-scheme="slate"] {
  --md-hue: 215;
  --md-default-bg-color: #0f1621;
  --md-primary-fg-color: #0b1119;
  --md-primary-fg-color--dark: #070b11;
  --md-code-bg-color: #18212e;
  --md-typeset-a-color: #ff8a4c;
  --md-accent-fg-color: #ff8a4c;
}
.md-header, .md-tabs { background-color: var(--md-primary-fg-color); }
.md-tabs__link--active, .md-tabs__item--active .md-tabs__link { color: var(--sh-ember); }
.md-header__button.md-logo .md-icon, .md-header__button.md-logo svg { fill: var(--sh-ember); }
.md-typeset h1 { font-weight: 700; letter-spacing: -0.02em; }
.md-typeset .grid.cards > ul > li { border-radius: 0.6rem; transition: border-color 0.15s; }
.md-typeset .grid.cards > ul > li:hover { border-color: var(--md-accent-fg-color); }
.md-typeset table:not([class]) { font-size: 0.72rem; }
.doc-signature { border-radius: 0.4rem; border-left: 3px solid var(--sh-ember); }
.md-typeset .admonition.tip, .md-typeset details.tip { border-color: var(--sh-ember); }
.md-typeset .tip > .admonition-title, .md-typeset .tip > summary {
  background-color: var(--md-accent-fg-color--transparent);
}
.md-typeset .tip > .admonition-title::before, .md-typeset .tip > summary::before {
  background-color: var(--sh-ember);
}
"""


# --------------------------------------------------------------------------- main


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    features = parse_roadmap()
    public = public_members()
    examples = parse_index()
    for ex in examples:
        load_example(ex, public)

    pages = {
        "README.md": "index.md",
        "ROADMAP.md": "roadmap.md",
        "DESIGN.md": "design.md",
        "examples/README.md": "guides/index.md",
    }
    pages |= {f"examples/{ex.path.name}": f"guides/{ex.slug}.md" for ex in examples}
    site = Site(out, pages)

    sections: dict[str, list[Example]] = {}
    for ex in examples:
        sections.setdefault(ex.section, []).append(ex)
        site.write(f"guides/{ex.slug}.md", guide_page(ex, features, public))

    site.write("index.md", home_page(site, sections))
    site.write(
        "guides/index.md",
        site.rewrite((EXAMPLES / "README.md").read_text(), "examples/README.md", "guides/index.md"),
    )
    for name in ("ROADMAP", "DESIGN"):
        site.write(
            pages[f"{name}.md"],
            site.rewrite((ROOT / f"{name}.md").read_text(), f"{name}.md", pages[f"{name}.md"]),
        )
    site.write("cli.md", cli_page(examples))
    site.write("reference/index.md", reference_index(public))

    all_modules = sorted(
        f"skyhammer.{p.stem}" for p in SRC.glob("*.py") if p.stem not in SKIP_MODULES
    )
    # Framework adapters are imported from their own modules, not from ``skyhammer``.
    adapters = sorted(
        f"skyhammer.integrations.{p.stem}"
        for p in (SRC / "integrations").glob("*.py")
        if not p.stem.startswith("_")
    )
    all_modules += adapters
    modules = sorted({m for m, _ in public.values() if m in all_modules} | set(adapters))
    internals = [m for m in all_modules if m not in modules]
    for module in all_modules:
        site.write(ref_page(module), reference_page(module, examples, public))
    site.write("assets/extra.css", EXTRA_CSS)
    site.prune()
    _write_if_changed(out.parent / "mkdocs.yml", mkdocs_yml(out, sections, modules, internals))
    print(f"generated {sum(1 for _ in out.rglob('*.md'))} pages in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build" / "docs" / "src")
