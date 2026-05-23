# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is **Andrew's Notebook** - a technical documentation site hosted on GitHub Pages covering science and technology topics. It's a **single-maintainer project** by @AndrewAltimit:

- **GitHub Pages Site**: https://andrewaltimit.github.io/Documentation/
- **Content Focus**: Physics (quantum mechanics, relativity, thermodynamics) and Technology (Docker, Kubernetes, AWS, Git)
- **Jekyll-based**: Uses the minimal-mistakes theme with professional wiki-style presentation
- **Self-hosted CI/CD**: All workflows run on self-hosted GitHub Actions runners

This repository is a pure documentation site. The shared automation tooling (link
checkers, CI helpers, agent integrations) lives in the `template-repo` and is
installed on the self-hosted runner, so it does not need to be vendored here.

## Site Structure

```
Documentation/
├── github-pages/          # Jekyll site source
│   ├── docs/             # Main documentation pages (physics/, technology/, ai-ml/, ...)
│   ├── code-examples/    # Code referenced by the docs
│   ├── _data/            # Navigation and site data
│   ├── _includes/        # Reusable HTML partials
│   ├── _layouts/         # Custom Jekyll layouts
│   ├── assets/           # CSS, JS, images
│   └── images/           # Site images
├── _config.yml           # Jekyll config (copied into github-pages/ at build time)
├── Gemfile               # Ruby/Jekyll dependencies
├── assets/css/           # Source CSS copied into github-pages/ at build time
└── images/               # Shared images referenced by docs
```

## Commands

### Building the Site

```bash
# Build the site as CI does it: assemble config + assets into github-pages/, then build
cp _config.yml Gemfile Gemfile.lock github-pages/ 2>/dev/null || true
mkdir -p github-pages/assets/css && cp -r assets/css/* github-pages/assets/css/

cd github-pages
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  jekyll/jekyll:4.2.2 \
  /bin/bash -c "bundle install && jekyll build"

# Serve locally for testing (http://localhost:4000)
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  -p 4000:4000 \
  jekyll/jekyll:4.2.2 \
  jekyll serve --host 0.0.0.0
```

### Link Checking

Links are validated with **html-proofer** against the built `_site` (not the raw
`.md` sources), so it understands Jekyll's `.md`→`.html` permalinks. It runs as
part of the Jekyll build container:

```bash
cd github-pages
bundle exec jekyll build
bundle exec htmlproofer ./_site \
  --checks Links --disable-external --no-enforce-https \
  --swap-urls '^/Documentation/:/'
```

`--swap-urls` strips the `/Documentation` baseurl so root-relative links resolve;
`--checks Links` scopes it to link/anchor integrity (not image alt-text or
HTTPS-enforcement). Note: the site's `advanced/` pages use directory-style
permalinks (`/docs/advanced/foo/`), so links to them must be `../foo/`, and links
out of them need an extra `../`.

## GitHub Actions Integration

Two workflows, both on self-hosted runners:

- **`pr-validation.yml`** — on PRs: builds the site and runs html-proofer link
  validation against the rendered `_site`. No AI/agent code review runs in this
  pipeline.
- **`jekyll.yml`** — on push to `main`/`update-docs`: builds the site and deploys
  the output to the `gh-pages` branch.

## Development Reminders

- **Content Focus**: This is a documentation site - focus on clear, accurate technical writing
- **Wiki Style**: Professional reference documentation, not tutorials
- **Visual Elements**: Use diagrams and code examples where helpful
- **Link Integrity**: Always verify internal links when moving/renaming pages
- After editing content, build the site locally (above) to catch Liquid/front-matter errors
- NEVER commit changes unless the user explicitly asks you to

## GitHub Etiquette

**IMPORTANT**: When working with GitHub issues, PRs, and comments:

- **NEVER use @ mentions** unless referring to actual repository maintainers
- Only @ mention the repository owner (@AndrewAltimit)

### PR Comments and Reactions

**Use Custom Reaction Images**: When commenting on PRs and issues, use our custom reaction images.

- **Available reactions**: https://raw.githubusercontent.com/AndrewAltimit/Media/refs/heads/main/reaction/config.yaml
- **Format**: `![Reaction](https://raw.githubusercontent.com/AndrewAltimit/Media/refs/heads/main/reaction/[filename])`

**CRITICAL: Proper Method for GitHub Comments with Reaction Images**

When posting PR/issue comments with reaction images:

1. **Use the Write tool** to create a temporary markdown file (e.g., `/tmp/comment.md`)
2. Use `gh pr comment --body-file /tmp/filename.md` to post the comment

**DO NOT USE** direct `--body` flag, heredocs, or echo commands - they will escape the `!` in `![Reaction]`.

## Documentation Topics

### Physics
- Classical Mechanics
- Quantum Mechanics
- Quantum Field Theory
- Relativity (Special and General)
- Statistical Mechanics
- Thermodynamics
- Condensed Matter Physics
- String Theory

### Technology
- Docker containerization
- Kubernetes orchestration
- Terraform infrastructure as code
- AWS cloud services
- Git version control
- Branching strategies
- Networking fundamentals
- Database design
- Cybersecurity basics

### AI/ML
- Stable Diffusion fundamentals
- ComfyUI workflows
- LoRA training concepts
- Base model comparisons
- ControlNet techniques
