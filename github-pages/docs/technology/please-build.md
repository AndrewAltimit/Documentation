---
layout: docs
title: Please Build
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Please Build</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">High-performance polyglot build system for monorepos</p>
</div>


<div class="intro-card">
  <p class="lead-text">Please (the <code>plz</code> command) is a high-performance, extensible build system that brings the power of Google's Blaze/Bazel to a wider audience with a more approachable syntax and philosophy. Designed for polyglot environments and monorepos, Please emphasizes correctness, reproducibility, and speed. This guide covers everything from basic setup to advanced remote execution.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-project-diagram"></i>
    <h4>Build Graph</h4>
    <p>Targets declare inputs; Please rebuilds only what changed</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-cube"></i>
    <h4>Hermetic &amp; Cached</h4>
    <p>Isolated, content-addressed builds that are reproducible everywhere</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-language"></i>
    <h4>Polyglot</h4>
    <p>One consistent build across Go, Python, Java, C++, Rust, and more</p>
  </div>
</div>

<div class="tip-card">
  <h4>When does a tool like Please earn its keep?</h4>
  <p>For a single-language app, your language's native tool (<code>go build</code>, <code>npm</code>, <code>cargo</code>) is simpler. Please shines in <strong>polyglot monorepos</strong> where you need one consistent, cached, parallel build across many languages — and where reproducibility and incremental rebuilds across a large dependency graph actually matter.</p>
</div>

### How Please thinks: the build graph

Every target declares its inputs and dependencies in a `BUILD` file. Please assembles these into a directed graph, then builds only what changed — running independent branches in parallel and reusing cached results for everything else.

```mermaid
flowchart BT
    UTILS["//common:utils"] --> LIB["//src:lib"]
    LIB --> APP["//src:app (binary)"]
    LIB --> TEST["//src:lib_test"]
    REQ["//third_party/python:requests"] --> APP
    style APP fill:#4facfe,color:#fff
    style TEST fill:#00c9a7,color:#fff
```

Change `utils` and Please rebuilds `lib`, `app`, and `lib_test`; change only the test, and just the test reruns.

## Key Features

<div class="command-grid">
  <div class="nav-card">
    <h4><i class="fas fa-language"></i> Language Agnostic</h4>
    <p>First-class rules for Go, Python, Java, C++, JavaScript, and Rust, with custom rules for anything else.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-cube"></i> Hermetic Builds</h4>
    <p>Each action runs in an isolated sandbox, so the same inputs always produce the same outputs.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-bolt"></i> Parallel &amp; Incremental</h4>
    <p>Independent targets build concurrently; content-addressed caching rebuilds only what changed.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-network-wired"></i> Remote Execution</h4>
    <p>Fan builds out across a worker pool and share a remote cache for team-wide speedups.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-puzzle-piece"></i> Extensible</h4>
    <p>Write custom rules in <code>build_defs</code> using a Python-like dialect (Starlark-style).</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-diagram-project"></i> Queryable Graph</h4>
    <p>Inspect, visualize, and reason about dependencies with <code>plz query</code>.</p>
  </div>
</div>

### How it compares

Please occupies the same niche as Bazel and Buck — correct, cached, graph-driven builds — but trades some of Bazel's ecosystem breadth for a gentler learning curve.

| | Please | Bazel | Native tools (`go`/`npm`/`cargo`) |
|---|--------|-------|-----------------------------------|
| **Sweet spot** | Polyglot monorepos | Very large polyglot monorepos | Single-language projects |
| **Learning curve** | Moderate | Steep | Low |
| **Hermeticity** | Yes (sandboxed) | Yes (sandboxed) | No (relies on local env) |
| **Remote cache / exec** | Built-in (REAPI) | Built-in (REAPI) | None |
| **Rule language** | Python-like build defs | Starlark | N/A |

<div class="tip-card">
  <h4>Both Please and Bazel speak REAPI</h4>
  <p>Remote caching and execution use the <strong>Remote Execution API</strong> standard, so Please can share a cache/executor backend (such as BuildBarn or BuildBuddy) with other REAPI-compatible build tools.</p>
</div>

## Installation

### Quick Install (Recommended)

```bash
# Latest stable version
curl -sSfL https://get.please.build | bash

# Or pin a specific version (recommended for reproducibility)
curl -sSfL https://get.please.build | bash -s -- --version=17.8.0
```

<div class="tip-card">
  <h4>Pin the version</h4>
  <p>Pin a specific release here and in <code>.plzconfig</code> — check the <a href="https://github.com/thought-machine/please/releases">releases page</a> for the current one. Pinning is what keeps every machine on an identical <code>plz</code>.</p>
</div>

### Alternative Installation Methods

```bash
# macOS with Homebrew
brew tap thought-machine/please
brew install please

# From source
git clone https://github.com/thought-machine/please.git
cd please
./bootstrap.sh

# Using Go
go install github.com/thought-machine/please@latest
```

### Verify Installation

```bash
plz --version
# Output: Please version 17.8.0
```

## Getting Started

### Creating a New Project

```bash
# Initialize Please in the current repository
plz init
```

This creates:
- `.plzconfig` — the main configuration file at the repo root
- `pleasew` — a wrapper script that bootstraps the pinned Please version, so contributors and CI don't need Please pre-installed (commit it and run `./pleasew build //...`)

Language support is added through **plugins** rather than templates. Pull in the rules for a language with:

```bash
plz init plugin go
plz init plugin python
plz init plugin java
```

<div class="tip-card">
  <h4>Plugins, not templates</h4>
  <p>Older guides reference per-language <code>plz init --template=…</code> flags; current Please uses the plugin system above. Check <code>plz init --help</code> for the options your installed version supports.</p>
</div>

## Configuration

### Basic Configuration

Edit `.plzconfig` to configure Please Build:

```ini
[please]
version = 17.8.0
selfupdate = true
location = ~/.please

[build]
path = src/
languages = python,go,java
timeout = 600
workers = 4

[cache]
dir = ~/.cache/please
httpurl = https://cache.example.com  # Optional remote cache

[python]
defaultinterpreter = python3
piptool = pip3
moduledir = third_party/python

[go]
goroot = /usr/local/go
importpath = github.com/myorg/myproject
```

### Advanced Configuration

```ini
[remote]
url = grpc://remote-execution.example.com:8980
instancename = main
numexecutors = 100

[metrics]
pushgatewayurl = http://prometheus-pushgateway:9091

[experimental]
go_modules = true
python_wheel = true
rust_cargo = true
```

## Build Rules

### Core Concepts

Build rules define how to build targets. Create `BUILD` files (or `BUILD.plz`) in directories:

### Python Example

```python
# BUILD file
python_binary(
    name = "app",
    main = "main.py",
    deps = [
        ":lib",
        "//third_party/python:requests",
    ],
)

python_library(
    name = "lib",
    srcs = glob(["*.py"], exclude=["*_test.py", "main.py"]),
    deps = [
        "//common:utils",
    ],
)

python_test(
    name = "lib_test",
    srcs = ["lib_test.py"],
    deps = [":lib"],
)
```

### Go Example

```python
go_binary(
    name = "server",
    srcs = ["main.go"],
    deps = [
        ":handlers",
        "//third_party/go:github.com_gorilla_mux",
    ],
)

go_library(
    name = "handlers",
    srcs = glob(["*.go"], exclude=["*_test.go", "main.go"]),
    visibility = ["//service/..."],
)

go_test(
    name = "handlers_test",
    srcs = ["handlers_test.go"],
    deps = [":handlers"],
)
```

<div class="tip-card">
  <h4>Hermeticity gotcha</h4>
  <p>Because builds run in a sandbox, a target can only see files it explicitly declares as <code>srcs</code> or <code>deps</code>. A build that "works on my machine" but fails under Please is almost always reading an undeclared file. List every input — that strictness is exactly what makes the build reproducible.</p>
</div>

### Cross-Language Dependencies

```python
# Protocol buffers used by multiple languages
proto_library(
    name = "api_proto",
    srcs = ["api.proto"],
    languages = ["python", "go", "java"],
    visibility = ["PUBLIC"],
)

# Docker image with multi-language app
docker_image(
    name = "microservice",
    srcs = [
        ":go_server",
        ":python_worker",
    ],
    base = "alpine:3.18",
    dockerfile = "Dockerfile",
)
```

## Testing

### Writing Tests

Please Build has first-class support for testing:

```python
# Unit tests
python_test(
    name = "unit_tests",
    srcs = glob(["*_test.py"]),
    deps = [":lib"],
    size = "small",
)

# Integration tests
python_test(
    name = "integration_tests",
    srcs = ["integration_test.py"],
    deps = [":app"],
    size = "medium",
    timeout = 300,
    labels = ["integration"],
)

# Benchmarks
go_test(
    name = "bench",
    srcs = ["bench_test.go"],
    deps = [":lib"],
    flags = "-bench=.",
    labels = ["benchmark"],
)
```

### Running Tests

```bash
# Run all tests
plz test

# Run specific test
plz test //src:unit_tests

# Run tests matching pattern
plz test //..._test

# Run tests with specific label
plz test --include integration

# Run each test multiple times (e.g. to flush out flaky tests)
plz test //src:unit_tests --num_runs=10

# Generate coverage report
plz cover //src:unit_tests
```

### Test Sharding

```python
# Automatically shard large test suites
python_test(
    name = "large_test_suite",
    srcs = glob(["test_*.py"]),
    shard_count = 4,  # Split across 4 parallel jobs
)
```

## Continuous Integration

### GitHub Actions

```yaml
name: Please Build CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Please
      run: |
        curl -sSfL https://get.please.build | bash
        # Make plz available to every subsequent step (PATH set with `export`
        # only survives within a single `run:` block).
        echo "$HOME/.please/bin" >> "$GITHUB_PATH"
    
    - name: Build
      run: plz build //...
    
    - name: Test
      run: plz test //...
    
    - name: Coverage
      run: plz cover //... --coverage_results_file=cover.xml
    
    - uses: codecov/codecov-action@v4
      with:
        file: ./cover.xml
```

### GitLab CI

```yaml
image: ubuntu:22.04

before_script:
  - apt-get update && apt-get install -y curl
  - curl -sSfL https://get.please.build | bash
  - export PATH="$HOME/.please/bin:$PATH"

build:
  script:
    - plz build //...
  artifacts:
    paths:
      - plz-out/

test:
  script:
    - plz test //...
  artifacts:
    reports:
      # Please writes combined results in xUnit/JUnit format here by default.
      junit: plz-out/log/test_results.xml
```

<div class="tip-card">
  <h4>Pin the version in CI</h4>
  <p>The <code>version</code> field in <code>.plzconfig</code> makes Please self-bootstrap to that exact release on every machine, so the bootstrap script in CI and every developer's laptop all run the same <code>plz</code>. Combined with a shared remote cache, this is what makes "it builds the same everywhere" true rather than aspirational.</p>
</div>

### Remote Caching for CI

```ini
# .plzconfig for CI
[cache]
dir = ~/.cache/please
httpurl = https://please-cache.example.com
httpwriteable = true
httpheaders = Authorization: Bearer $CACHE_TOKEN
```

## Advanced Features

### Remote Execution

Distribute builds across multiple machines:

```ini
# .plzconfig
[remote]
url = grpc://remote.example.com:8980
instancename = main
numexecutors = 50
casurl = grpc://cas.example.com:8981
```

### Custom Build Rules

```python
# build_defs/BUILD
filegroup(
    name = "rules",
    srcs = ["rust_rules.build_defs"],
    visibility = ["PUBLIC"],
)
```

```python
# build_defs/rust_rules.build_defs
def rust_binary(name, srcs, deps=None, visibility=None):
    """Build a Rust binary."""
    return build_rule(
        name = name,
        srcs = srcs,
        deps = deps,
        outs = [name],
        cmd = "rustc $SRCS -o $OUT",
        binary = True,
        visibility = visibility,
    )
```

### Build Graph Analysis

```bash
# Visualize dependencies
plz query graph --to //src:app | dot -Tpng > graph.png

# Find all reverse dependencies
plz query revdeps //common:utils

# Query for specific attributes
plz query print //src:app --field=deps

# Find all tests
plz query alltargets --include test
```

### Performance Optimization

```ini
[build]
workers = 16  # Parallel build jobs
memorylimit = 8GB

[test]
defaulttimeout = 300
workers = 8

[metrics]
pushgatewayurl = http://prometheus:9091
namespace = please_build
```

### Integration with Modern Tools

#### Docker Support
```python
docker_image(
    name = "app_image",
    srcs = [":app_binary"],
    dockerfile = "Dockerfile",
    labels = ["latest", "$VERSION"],
    repo = "myorg/myapp",
)
```

#### Kubernetes Deployment
```python
k8s_config(
    name = "deployment",
    srcs = ["k8s/*.yaml"],
    containers = {
        "app": ":app_image",
    },
)
```

#### Protocol Buffers & gRPC
```python
grpc_library(
    name = "api_grpc",
    srcs = ["api.proto"],
    languages = ["python", "go"],
    protoc_flags = ["--experimental_allow_proto3_optional"],
)
```

## Best Practices

### Monorepo Organization

```
/
├── .plzconfig
├── BUILD              # Root build file
├── build_defs/        # Custom build rules
├── common/            # Shared libraries
├── services/          # Microservices
│   ├── api/
│   ├── auth/
│   └── worker/
├── tools/             # Development tools
└── third_party/       # External dependencies
    ├── go/
    ├── python/
    └── java/
```

### Dependency Management

```python
# third_party/python/BUILD
pip_library(
    name = "requests",
    version = "2.31.0",
    hashes = ["sha256:..."],
    deps = [
        ":urllib3",
        ":certifi",
    ],
)

# Lock dependencies
# Run: plz hash --update //third_party/python/...
```

### Build Optimization Tips

1. **Use Remote Caching**: Share build artifacts across team
2. **Minimize Dependencies**: Keep build graphs shallow
3. **Parallelize Tests**: Use test sharding for large suites
4. **Per-environment config**: keep CI-specific overrides in `.plzconfig.ci` and select it with `plz build --profile=ci //...`
5. **Incremental Builds**: Design rules for maximum incrementality

## Troubleshooting

### Common Issues

```bash
# Clean all cached outputs (forces a full rebuild next time)
plz clean

# Clean and rebuild just one target
plz clean //src:app && plz build //src:app

# Drop into a debugger for a failing test
plz test //src:app_test --debug

# Stream full subprocess output instead of Please's summary view
plz build //src:app --show_all_output

# Record a Chrome-tracing timeline of the build
plz build //src:app --trace_file=trace.json
```

### Build Reproducibility

Hermetic, content-addressed builds should be bit-for-bit reproducible: the same inputs produce the same output hash. You can verify this by building, clearing the cache, and rebuilding:

```bash
plz build //src:app
sha256sum plz-out/bin/src/app

plz clean
plz build //src:app
sha256sum plz-out/bin/src/app   # hash should match the first build
```

## Migration Guide

### From Bazel

Core rule names and the `//package:target` label syntax are deliberately close to Bazel's, so simple targets often port verbatim:

```python
# Bazel and Please both spell this the same way
cc_binary(
    name = "app",
    srcs = ["main.cc"],
    deps = [":lib"],
)
```

The real differences are in the surrounding ecosystem: Bazel's `WORKSPACE`/`MODULE.bazel` and `http_archive` become Please's `.plzconfig` plus per-language rules like `pip_library` and `go_module`, and Please's rule language is a Python-like dialect rather than strict Starlark. Expect to rewrite third-party dependency declarations rather than your own targets.

### From Make

```makefile
# Makefile
app: main.o lib.o
    gcc -o app main.o lib.o

# Please BUILD file
cc_binary(
    name = "app",
    srcs = ["main.c"],
    deps = [":lib"],
)
```

## FAQ

**Q: How does Please compare to Bazel?**
A: Please is inspired by Bazel but focuses on simplicity and ease of use. It has a gentler learning curve while maintaining most of Bazel's power.

**Q: Can I use Please for small projects?**
A: Yes! Please scales from single-file projects to massive monorepos.

**Q: Does Please support Windows?**
A: Please has experimental Windows support via WSL2.

**Q: How do I debug failing builds?**
A: Run with `--show_all_output` to see full subprocess logs, drop into a debugger on a failing test with `plz test //... --debug`, or inspect the per-target logs under `plz-out/log/`.

For more FAQs, see the [official FAQ](https://please.build/faq.html).

## Key Takeaways

<div class="takeaway-card">
  <ul>
    <li><strong>Please targets polyglot monorepos</strong> — one build system across Go, Python, Java, C++, and more, with a gentler learning curve than Bazel.</li>
    <li><strong>The build graph drives everything:</strong> declare inputs and deps in <code>BUILD</code> files, and Please rebuilds only what changed.</li>
    <li><strong>Content-addressed caching plus parallelism</strong> deliver fast, incremental builds; remote caching and execution scale this across a team.</li>
    <li><strong>Hermetic builds</strong> make results reproducible — the same inputs always produce the same outputs.</li>
    <li><strong>Use native tooling for single-language projects;</strong> reach for Please when scale, polyglot needs, or reproducibility justify it.</li>
  </ul>
</div>

## Resources

- [Official Documentation](https://please.build/)
- [GitHub Repository](https://github.com/thought-machine/please)
- [Rule Examples](https://github.com/thought-machine/please/tree/master/test)
- [Please Community Discussions](https://github.com/thought-machine/please/discussions)
- [Build Language Reference](https://please.build/language.html)
- [Please FAQ](https://please.build/faq.html) - Common questions and answers

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="ci-cd/">CI/CD</a> — wire Please builds into automated pipelines</li>
    <li><a href="git/">Git Version Control</a> — monorepo strategies and large-repo tooling</li>
    <li><a href="docker/">Docker</a> — package Please build artifacts into container images</li>
    <li><a href="kubernetes/">Kubernetes</a> — deploy the services Please builds</li>
  </ul>
</div>
