# Andrew's Notebook

Technical documentation repository hosting a GitHub Pages site covering physics, technology, and AI/ML topics.

**Live Site:** https://andrewaltimit.github.io/Documentation/

## Overview

This repository is the source for a public-facing technical wiki covering quantum
computing, AI/ML, containerization, cloud infrastructure, and more. It is built
with Jekyll (minimal-mistakes theme) and deployed to GitHub Pages by a self-hosted
GitHub Actions runner.

Shared CI/automation tooling (link checking, etc.) is maintained centrally and
installed on the self-hosted runner, so it is not vendored in this repository.

## Repository Structure

```
Documentation/
├── github-pages/          # Jekyll site source (docs, code-examples, layouts, assets)
├── _config.yml            # Jekyll config (assembled into github-pages/ at build time)
├── Gemfile                # Ruby/Jekyll dependencies
├── assets/css/            # Source CSS assembled into github-pages/ at build time
└── images/                # Shared images referenced by docs
```

## Quick Start

### Build the Site Locally

```bash
# Assemble config + assets, then build/serve from github-pages/
cp _config.yml Gemfile Gemfile.lock github-pages/ 2>/dev/null || true
mkdir -p github-pages/assets/css && cp -r assets/css/* github-pages/assets/css/

cd github-pages
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  -p 4000:4000 \
  jekyll/jekyll:4.2.2 \
  /bin/bash -c "bundle install && jekyll serve --host 0.0.0.0"
# Site at http://localhost:4000
```

## Continuous Integration

- **Pull requests** run an internal markdown link check and a Jekyll build smoke
  test on the self-hosted runner (`.github/workflows/pr-validation.yml`).
- **Pushes to `main`** build the site and deploy it to the `gh-pages` branch
  (`.github/workflows/jekyll.yml`).

## Documentation Topics

### Physics
- Quantum Mechanics and Quantum Field Theory
- Special and General Relativity
- Statistical Mechanics and Thermodynamics
- Condensed Matter Physics

### Technology
- Docker and Kubernetes
- Terraform and AWS
- Git and Version Control
- Database Design
- Cybersecurity

### AI/ML
- Stable Diffusion and Diffusion Models
- ComfyUI Workflows
- LoRA Training
- Transformer Architectures

## Contributing

Key files for contributors:
- `CLAUDE.md` - Instructions for the Claude Code AI assistant
- `github-pages/` - All site content and layout

## License

See [LICENSE](LICENSE) for details.

---

Maintained by [@AndrewAltimit](https://github.com/AndrewAltimit)
