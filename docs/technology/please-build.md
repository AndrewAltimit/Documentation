<link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css">

# Please Build

Please Build is a versatile build automation tool that simplifies the process of building, testing, and deploying your projects. This documentation provides you with a step-by-step guide to getting started with Please Build and integrating it into your workflow.

## Key Features

- **Language Agnostic**: Please Build supports numerous programming languages, making it versatile for various projects.
- **Flexible**: The tool offers a customizable configuration to cater to specific project requirements.
- **Parallel Execution**: Please Build can run tasks concurrently, speeding up the build process.
- **Incremental Builds**: The tool efficiently rebuilds only what has changed, saving time and resources.
- **Extensible**: Please Build allows developers to create custom build definitions for unique scenarios.

## Installation

To install Please Build, run the following command in your terminal:

```bash
curl -sSfL https://get.please.build | bash
```

## Getting Started

To create a new Please Build project, navigate to your project directory and run the following command:

```bash
plz init
```

This will generate a `plzconfig` file in the root directory, which will be used to configure Please Build.

## Configuration

Edit the `plzconfig` file to configure Please Build according to your project's requirements. Below is a sample `plzconfig` configuration:

```c
[please]
version = 17.0.0
selfupdate = true

[build]
path = src/
languages = python,go
```

For more details on the available configuration options, refer to the [official documentation](https://please.build/configuration.html).

## Build Rules

Please Build uses build rules to specify how to build, test, and deploy your project. Create a `BUILD.plz` file in the appropriate directory and define your build rules. For example:

```c
python_binary(
    name = "main",
    main = "main.py",
    deps = [":lib"],
)

python_library(
    name = "lib",
    srcs = glob(["*.py"]),
)
```

For more information on build rules and how to customize them, check out the [build rules documentation](https://please.build/rules.html).

## Testing

Please Build supports many popular testing frameworks. To add a test rule to your project, include it in the `BUILD.plz` file. Here's an example of a Python test rule:

```c
python_test(
    name = "tests",
    srcs = glob(["*_test.py"]),
    deps = [":lib"],
)
```

To run tests, use the following command:

```bash
plz test
```

## Continuous Integration

Integrating Please Build with continuous integration (CI) services is simple. Add the following configuration to your CI service's configuration file (e.g., `.travis.yml` for Travis CI):

```yaml
language: minimal

install:
  - curl -sSfL https://get.please.build | bash

script:
  - plz build
  - plz test
```

## FAQ

For frequently asked questions, please refer to the [FAQ section](https://please.build/faq.html) in the official documentation.
