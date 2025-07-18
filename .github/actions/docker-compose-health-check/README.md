# Docker Compose Health Check Action

A reusable composite GitHub Action for starting Docker Compose services and waiting for them to be healthy. This action streamlines the process of setting up containerized services in CI/CD workflows with proper health checking and error handling.

## Overview

This composite action provides a robust way to:
- Validate Docker Compose configuration
- Start services with optional image building
- Wait for services to become healthy via HTTP health checks
- Automatically clean up on failure
- Provide clear status feedback with emojis

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `services` | Space-separated list of services to start. Leave empty to start all services. | No | `''` (all services) |
| `health-endpoint` | HTTP endpoint URL to check for service health | No | `http://localhost:8005/health` |
| `timeout` | Maximum time to wait for services to be healthy (in seconds) | No | `60` |
| `build` | Whether to build Docker images before starting services | No | `true` |

## Usage Examples

### Basic Usage

Start specific services with health check:

```yaml
- uses: ./.github/actions/docker-compose-health-check
  with:
    services: 'mcp-server'
    health-endpoint: 'http://localhost:8005/health'
    timeout: '60'
    build: 'true'
```

### Start All Services

Start all services defined in docker-compose.yml:

```yaml
- uses: ./.github/actions/docker-compose-health-check
  with:
    services: ''  # empty string starts all services
    health-endpoint: 'http://localhost:8005/health'
```

### Skip Image Building

Use existing images without rebuilding (faster for repeated runs):

```yaml
- uses: ./.github/actions/docker-compose-health-check
  with:
    services: 'mcp-server'
    health-endpoint: 'http://localhost:8005/health'
    timeout: '30'
    build: 'false'
```

### Multiple Services

Start multiple services:

```yaml
- uses: ./.github/actions/docker-compose-health-check
  with:
    services: 'mcp-server python-ci'
    health-endpoint: 'http://localhost:8005/health'
    timeout: '90'
```

## Complete Workflow Example

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: ./.github/actions/docker-compose-health-check
        with:
          services: 'mcp-server'
          health-endpoint: 'http://localhost:8005/health'
          timeout: '60'
      
      - name: Run integration tests
        run: |
          # Your tests here
          docker-compose run --rm python-ci pytest tests/integration/
      
      - name: Clean up
        if: always()
        run: docker-compose down
```

## How It Works

1. **Configuration Validation**: Validates the docker-compose.yml file syntax
2. **Service Startup**: Starts services with optional `--build` flag
3. **Health Check Loop**: 
   - Polls the health endpoint every 2 seconds
   - Continues until successful response or timeout
   - Shows progress with dots while waiting
4. **Success/Failure Handling**:
   - On success: Displays service status
   - On timeout: Shows logs, stops services, and exits with error
5. **Automatic Cleanup**: Stops services on failure (via `if: failure()`)

## Features

- **Visual Feedback**: Uses emojis (🐳, ⏳, ✅, ❌, 🧹) for clear status indication
- **Debugging Support**: Automatically shows container logs on failure
- **Graceful Cleanup**: Ensures services are stopped even if health checks fail
- **Flexible Configuration**: All parameters are optional with sensible defaults
- **Error Handling**: Proper exit codes and informative error messages

## Requirements

- Docker and Docker Compose must be installed on the runner
- The health endpoint must return a successful HTTP response (2xx status code) when services are ready
- Services should be configured with appropriate health checks in docker-compose.yml

## Best Practices

1. **Use Specific Services**: Only start the services you need for better performance
2. **Adjust Timeouts**: Set appropriate timeouts based on service startup time
3. **Health Check Design**: Ensure your health endpoint accurately reflects service readiness
4. **Cleanup**: Always include a cleanup step in your workflow:
   ```yaml
   - name: Clean up
     if: always()
     run: docker-compose down
   ```

## Troubleshooting

- **Timeout Issues**: Increase the timeout value or optimize service startup
- **Health Check Failures**: Verify the health endpoint URL and that the service implements it correctly
- **Build Failures**: Check Docker image configurations and available resources
- **Port Conflicts**: Ensure required ports are available on the runner

## Migration Guide

To migrate from manual Docker Compose commands in workflows:

### Before:
```yaml
- name: Start services
  run: |
    docker-compose up -d
    sleep 30  # Hope services are ready
```

### After:
```yaml
- uses: ./.github/actions/docker-compose-health-check
  with:
    services: ''
    health-endpoint: 'http://localhost:8005/health'
```

This provides deterministic waiting with proper error handling instead of arbitrary sleep times.
