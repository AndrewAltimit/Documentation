# MCP Tools Documentation

This document provides detailed information about the containerized MCP (Model Context Protocol) tools in this project. Updated for 2024 with the latest tool versions and best practices.

## Container-First Design

All MCP tools run in Docker containers as part of this project's philosophy:

- **Zero local dependencies** - just Docker
- **Consistent execution** - same results on any Linux system with Python 3.11
- **Easy deployment** - works identically on self-hosted runners
- **Single maintainer friendly** - no complex setup or coordination needed
- **User permission handling** - containers run as current user to avoid permission issues

## Table of Contents

- [Overview](#overview)
- [Core Tools](#core-tools)
- [AI Integration Tools](#ai-integration-tools)
- [Content Creation Tools](#content-creation-tools)
- [Remote Services](#remote-services)
- [Custom Tool Development](#custom-tool-development)

## Overview

MCP tools are functions that can be executed through the MCP server to perform various development and content creation tasks. They are accessible via HTTP API or through the MCP protocol.

**Server Details:**
- **Framework**: FastAPI (0.100+) with Python 3.11+
- **Port**: 8005 (containerized)
- **Protocol**: HTTP/REST and MCP v2
- **Container**: Runs in `mcp-server` Docker container
- **Async**: Full async/await support with asyncio

### Tool Execution

All tools can be executed via POST request to `/tools/execute`:

```bash
curl -X POST http://localhost:8005/tools/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "tool": "tool_name",
    "arguments": {
      "arg1": "value1",
      "arg2": "value2"
    }
  }'

# With timeout for long-running tools
curl -X POST http://localhost:8005/tools/execute \
  -H "Content-Type: application/json" \
  --max-time 300 \
  -d '{"tool": "tool_name", "arguments": {}}'
```

## Core Tools

### format_check

Check code formatting according to language-specific standards.

**Parameters:**

- `path` (string): Path to the file or directory to check
- `language` (string): Programming language (python, javascript, typescript, go, rust)

**Example:**

```python
{
  "tool": "format_check",
  "arguments": {
    "path": "./src",
    "language": "python"
  }
}
```

**Response:**

```json
{
  "formatted": true,
  "output": "All files formatted correctly",
  "files_checked": 15,
  "execution_time": 1.23
}
```

### lint

Run static code analysis to find potential issues.

**Parameters:**

- `path` (string): Path to analyze
- `config` (string, optional): Path to linting configuration file

**Example:**

```python
{
  "tool": "lint",
  "arguments": {
    "path": "./src",
    "config": ".flake8"
  }
}
```

**Response:**

```json
{
  "success": true,
  "issues": [
    "src/main.py:10:1: E302 expected 2 blank lines, found 1"
  ],
  "severity_counts": {
    "error": 0,
    "warning": 1,
    "info": 3
  }
}
```

### Running CI/CD Pipeline

While not a direct MCP tool, the full CI/CD pipeline can be executed via:

```bash
# Run complete CI pipeline
./scripts/run-ci.sh full

# Individual stages
./scripts/run-ci.sh format
./scripts/run-ci.sh lint-basic
./scripts/run-ci.sh lint-full
./scripts/run-ci.sh security
./scripts/run-ci.sh test
```

These scripts leverage the containerized Python CI environment.

## AI Integration Tools

### consult_gemini

Get AI assistance from Google's Gemini model for technical questions, code review, and suggestions.

**Parameters:**

- `question` (string): The question or request
- `context` (string, optional): Additional context or code

**Example:**

```python
{
  "tool": "consult_gemini",
  "arguments": {
    "question": "How can I optimize this function for better performance?",
    "context": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
  }
}
```

**Response:**

```json
{
  "response": "The current implementation has exponential time complexity. Here's an optimized version using memoization...",
  "model": "gemini-2.5-pro",
  "tokens_used": 245,
  "suggestions": [
    {
      "type": "optimization",
      "description": "Use memoization to cache results",
      "code": "@functools.lru_cache(maxsize=None)\ndef fibonacci(n):..."
    }
  ]
}
```

### clear_gemini_history

Clear Gemini's conversation history to ensure fresh responses without cached context.

**Parameters:**

- None

**Example:**

```python
{
  "tool": "clear_gemini_history",
  "arguments": {}
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Cleared 5 conversation entries",
  "cleared_entries": 5
}
```

**Use Cases:**

- **Automatically called before PR reviews** to ensure fresh analysis
- When switching between different contexts
- To reset after errors or incorrect responses
- Prevents bias from previous conversations

### Advanced Gemini Features

The Gemini integration supports several specialized functions:

1. **Code Analysis**

   ```python
   result = await gemini.analyze_code(
       code=code_snippet,
       language="python",
       focus=["security", "performance", "best-practices"]
   )
   ```

2. **Error Explanation**

   ```python
   explanation = await gemini.explain_error(
       error_message=str(exception),
       code_context=surrounding_code,
       stack_trace=traceback.format_exc()
   )
   ```

3. **Documentation Generation**

   ```python
   docs = await gemini.generate_documentation(
       code=function_code,
       style="google",  # or "numpy", "sphinx"
       include_examples=True
   )
   ```

4. **Test Suggestion**

   ```python
   tests = await gemini.suggest_tests(
       code=function_code,
       framework="pytest",
       coverage_target=0.90,
       include_edge_cases=True
   )
   ```

## Content Creation Tools

### create_manim_animation

Create mathematical and technical animations using Manim.

**Parameters:**

- `script` (string): Manim Python script
- `output_format` (string): Output format (mp4, gif, webm)

**Example:**

```python
{
  "tool": "create_manim_animation",
  "arguments": {
    "script": "from manim import *\n\nclass Example(Scene):\n    def construct(self):\n        text = Text('Hello, MCP!')\n        self.play(Write(text))",
    "output_format": "mp4"
  }
}
```

**Response:**

```json
{
  "success": true,
  "output_path": "/app/output/manim/Example.mp4",
  "format": "mp4",
  "duration": 5.2,
  "resolution": "1920x1080",
  "file_size_mb": 2.3
}
```

### compile_latex

Compile LaTeX documents to various formats.

**Parameters:**

- `content` (string): LaTeX document content
- `format` (string): Output format (pdf, dvi, ps)

**Example:**

```python
{
  "tool": "compile_latex",
  "arguments": {
    "content": "\\documentclass{article}\\begin{document}\\title{Test}\\maketitle\\end{document}",
    "format": "pdf"
  }
}
```

**Response:**

```json
{
  "success": true,
  "output_path": "/app/output/latex/document_12345.pdf",
  "format": "pdf",
  "pages": 3,
  "compile_time": 1.8,
  "warnings": []
}
```

## Remote Services

### ComfyUI Integration

Access ComfyUI workflows for image generation.

**Setup Instructions:**

For detailed setup instructions, see the [ComfyUI MCP Server Setup Guide](https://gist.github.com/AndrewAltimit/f2a21b1a075cc8c9a151483f89e0f11e).

**Available Tools:**

- `generate_image`: Generate images using workflows
- `list_workflows`: List available workflows
- `execute_workflow`: Execute specific workflow with custom parameters
- `get_workflow_info`: Get detailed workflow information
- `list_models`: List available models (checkpoints, LoRAs, VAEs)
- `queue_status`: Check generation queue status

**Configuration:**

```bash
COMFYUI_SERVER_URL=http://192.168.0.152:8189
```

### AI Toolkit Integration

Train LoRA models using AI Toolkit.

**Setup Instructions:**

For detailed setup instructions, see the [AI Toolkit MCP Server Setup Guide](https://gist.github.com/AndrewAltimit/2703c551eb5737de5a4c6767d3626cb8).

**Available Tools:**

- `upload_dataset`: Upload training dataset with validation
- `create_training_config`: Configure training parameters
- `start_training`: Begin training job with queue management
- `check_training_status`: Monitor progress with ETA
- `list_models`: List trained models with metadata
- `download_model`: Download trained model with optional metadata
- `cancel_training`: Cancel running training job
- `get_training_logs`: Retrieve training logs

**Configuration:**

```bash
AI_TOOLKIT_SERVER_URL=http://192.168.0.152:8190
```

## Custom Tool Development

### Creating a New Tool

1. **Define the tool function in mcp_server.py:**

```python
# tools/mcp/mcp_server.py
from typing import Dict, Any, Optional
import asyncio

class MCPTools:
    @staticmethod
    async def my_custom_tool(
        param1: str, 
        param2: int = 10,
        timeout: Optional[float] = 30.0
    ) -> Dict[str, Any]:
        """
        My custom tool description.

        Args:
            param1: Description of param1
            param2: Description of param2
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with results
            
        Raises:
            TimeoutError: If execution exceeds timeout
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not param1:
            raise ValueError("param1 cannot be empty")
            
        # Tool implementation with timeout
        try:
            result = await asyncio.wait_for(
                process_data(param1, param2),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool execution exceeded {timeout}s")

        return {
            "success": True,
            "result": result,
            "metadata": {
                "param1": param1,
                "param2": param2,
                "execution_time": time.time() - start_time
            }
        }
```

2. **Handle in execute_tool endpoint:**

```python
# tools/mcp/mcp_server.py
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

@app.post("/tools/execute")
async def execute_tool(request: ToolRequest):
    try:
        # ... existing code
        elif tool_name == "my_custom_tool":
            result = await MCPTools.my_custom_tool(**request.arguments)
            
        # Log successful execution
        logger.info(f"Tool {tool_name} executed successfully")
        return result
        
    except ValueError as e:
        logger.error(f"Invalid parameters for {tool_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        logger.error(f"Timeout executing {tool_name}: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

3. **Update configuration:**

```json
// .mcp.json
{
  "tools": {
    "my_custom_tool": {
      "description": "My custom tool description",
      "parameters": {
        "param1": {
          "type": "string",
          "description": "Description of param1",
          "required": true
        },
        "param2": {
          "type": "integer",
          "description": "Description of param2",
          "default": 10
        }
      }
    }
  }
}
```

### Tool Guidelines

1. **Error Handling:**
   - Always return structured responses
   - Include error details in response
   - Use try-except blocks
   - Log errors for debugging

2. **Async Operations:**
   - Use `async/await` for I/O operations
   - Handle timeouts appropriately
   - Consider concurrent execution
   - Mock subprocess calls in tests

3. **Input Validation:**
   - Validate all parameters
   - Provide helpful error messages
   - Use type hints (Python 3.11 features)
   - Sanitize file paths

4. **Output Format:**
   - Return consistent structure
   - Include success status
   - Provide meaningful metadata
   - Use proper JSON serialization

### Testing Tools

```python
# tests/test_custom_tool.py
import pytest
from tools.mcp.custom_tools import my_custom_tool
import asyncio

@pytest.mark.asyncio
async def test_my_custom_tool():
    """Test basic functionality of custom tool."""
    result = await my_custom_tool("test", 42)

    assert result["success"] is True
    assert result["result"] is not None
    assert result["metadata"]["param1"] == "test"
    assert result["metadata"]["param2"] == 42
    assert "execution_time" in result["metadata"]

@pytest.mark.asyncio
async def test_my_custom_tool_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="param1 cannot be empty"):
        await my_custom_tool("", 42)

@pytest.mark.asyncio
async def test_my_custom_tool_timeout():
    """Test timeout handling."""
    with pytest.raises(TimeoutError):
        await my_custom_tool("test", 42, timeout=0.001)
```

## Best Practices

### Performance

1. **Caching:**
   - Cache expensive operations with `functools.lru_cache` or Redis
   - Use Redis for distributed caching across containers
   - Set appropriate TTL values (5 minutes for dynamic, 1 hour for static)
   - Implement cache warming for critical paths

2. **Concurrency:**
   - Use asyncio for I/O-bound operations
   - Implement rate limiting with `slowapi` or custom middleware
   - Handle concurrent requests with semaphores
   - Use connection pooling for external services
   - Configure appropriate worker counts (2 * CPU cores + 1)

### Security

1. **Input Sanitization:**
   - Validate all user inputs with Pydantic models
   - Escape special characters for shell commands
   - Limit resource usage (memory, CPU, execution time)
   - Use parameterized queries for any database operations
   - Implement request size limits

2. **Authentication:**
   - Implement API key authentication with rotation
   - Use secure communication (HTTPS with TLS 1.3)
   - Log access attempts with rate limiting per key
   - Consider OAuth2 for external integrations
   - Implement CORS policies for web access

### Monitoring

1. **Logging:**
   - Log all tool executions with structured logging (JSON)
   - Include timing information and request IDs
   - Track error rates and alert on thresholds
   - Use log rotation to prevent disk filling
   - Implement log aggregation for multi-container setup

2. **Metrics:**
   - Monitor tool usage
   - Track response times
   - Alert on failures

## Troubleshooting

### Common Issues

1. **Tool not found:**
   - Check tool registration in MCP server
   - Verify configuration file
   - Restart MCP server

2. **Timeout errors:**
   - Increase timeout in configuration
   - Optimize tool performance
   - Check network connectivity

3. **Permission errors:**
   - Verify file permissions
   - Check Docker volume mounts
   - Review security settings

### Debug Mode

Enable debug logging:

```bash
# Set environment variables
export LOG_LEVEL=DEBUG
export PYTHONFAULTHANDLER=1
export ASYNCIO_DEBUG=1

# Run with debug mode
docker-compose up mcp-server

# Or use the debug configuration
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up mcp-server
```

View logs:

```bash
docker-compose logs -f mcp-server
```

## API Reference

### Endpoints

- `GET /` - Server information
- `GET /health` - Health check
- `GET /tools` - List available tools
- `POST /tools/execute` - Execute a tool
- `GET /tools/{tool_name}` - Get tool details

### Response Format

```json
{
  "success": true,
  "result": {
    // Tool-specific results
  },
  "error": null,
  "metadata": {
    "tool": "tool_name",
    "execution_time": 1.23,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Error Codes

- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing or invalid API key)
- `403` - Forbidden (rate limit exceeded)
- `404` - Tool not found
- `408` - Request timeout
- `413` - Payload too large
- `429` - Too many requests
- `500` - Internal server error
- `503` - Service unavailable
- `504` - Gateway timeout (tool execution timeout)
