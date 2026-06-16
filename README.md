# DINO Industrial Defect Research

This repository tracks the DINOv3-based industrial defect research workflow.

The public documentation site is generated from `docs/` with MkDocs and deployed
to GitHub Pages. Internal raw notes live under `research_ops/` and are not part
of the published site.

## Documentation Workflow

1. Keep raw lab discussion notes in `research_ops/internal_discussions/` or
   `research_ops/raw_notes/`.
2. Extract only public-safe summaries into `docs/`.
3. Update experiment results in `docs/experiments/registry.md`.
4. Update technical decisions in `docs/decisions/decision-log.md`.
5. Push to `main`; GitHub Actions builds and deploys the Pages site.

## Local Preview

Install the documentation dependencies once:

```bash
python -m pip install -r requirements-docs.txt
```

Preview the site:

```bash
mkdocs serve
```

Build the site locally:

```bash
mkdocs build --strict
```
