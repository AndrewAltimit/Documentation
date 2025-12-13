# Setting Up Gemini AI Code Review

This repository includes automatic AI-powered code review for pull requests using Google's Gemini AI CLI. Updated for 2024 with the latest Gemini models and features.

## Features

- Automatic code review on every pull request
- **Conversation history cleared before each review** for fresh, unbiased analysis
- Analyzes code changes and provides constructive feedback
- Posts review comments directly to the PR
- Non-blocking - won't fail your PR if the CLI is unavailable
- Uses official Gemini CLI with automatic authentication
- Receives project-specific context from PROJECT_CONTEXT.md

## Setup Instructions

### For GitHub-Hosted Runners

The workflow will attempt to install Gemini CLI automatically if Node.js is available.

### For Self-Hosted Runners

1. **Install Node.js 18+** (recommended version 22.16.0 or latest LTS)

   ```bash
   # Using nvm (recommended)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   source ~/.bashrc
   nvm install 22.16.0
   nvm use 22.16.0
   nvm alias default 22.16.0
   ```

2. **Install Gemini CLI**

   ```bash
   npm install -g @google/gemini-cli
   
   # Verify installation
   gemini --version
   ```

3. **Authenticate** (happens automatically on first use)

   ```bash
   # Run the gemini command - it will open browser for OAuth authentication
   gemini
   
   # Test authentication
   echo "Hello, Gemini!" | gemini
   ```

That's it! The next time you open a pull request, Gemini will automatically review your code.

## How It Works

1. When a PR is opened or updated, the Gemini review job runs
2. **Conversation history is automatically cleared** using the `clear_gemini_history` MCP tool to ensure fresh, unbiased review
3. **Project context is loaded** from PROJECT_CONTEXT.md
4. It analyzes:
   - Project-specific context and philosophy
   - Changed files
   - Code diff
   - PR title and description
5. Gemini provides feedback on:
   - Container configurations and security
   - Code quality (with project standards in mind)
   - Potential bugs and edge cases
   - Project-specific concerns (from PROJECT_CONTEXT.md)
   - Positive aspects and improvements
   - Python 3.11+ compatibility
   - Async/await patterns
6. The review is posted as a comment on the PR

### Why Clear History?

Clearing conversation history before each review ensures:
- No bias from previous reviews
- Fresh perspective on each PR
- Consistent quality of feedback
- No confusion from unrelated context

## Project Context

Gemini receives detailed project context from `PROJECT_CONTEXT.md`, which includes:

- Container-first philosophy
- Single-maintainer design
- What to prioritize in reviews
- Project-specific patterns and standards

This ensures Gemini "hits the ground running" with relevant, actionable feedback.

## CLI Usage

The Gemini CLI can be used directly:

```bash
# Basic usage
echo "Your question here" | gemini

# Specify a model (2024 models)
echo "Technical question" | gemini -m gemini-2.5-pro
echo "Quick task" | gemini -m gemini-2.5-flash

# With file input
gemini < code_file.py

# With system prompt
echo "Review this code" | gemini --system "You are a senior Python developer"
```

## Rate Limits

Free tier limits (as of 2024):

- 60 requests per minute (RPM)
- 1,500 requests per day (RPD)
- 4 million tokens per day
- 10 million tokens per day for gemini-2.5-flash

For most single-maintainer projects, these limits are more than sufficient. The gemini-2.5-flash model offers higher limits for quick tasks.

## Customization

You can customize the review behavior by editing `scripts/gemini-pr-review.py`:

- Adjust the prompt to focus on specific aspects
- Change the model (default tries gemini-2.5-pro, falls back to gemini-2.5-flash)
- Modify comment formatting
- Add custom checks for your project
- Configure review depth and focus areas
- Set custom temperature for creativity (0.0-1.0)

## Troubleshooting

If Gemini reviews aren't working:

1. **Check Node.js version**: `node --version` (must be 18+, recommend 22+)
2. **Verify Gemini CLI installation**: `which gemini` and `gemini --version`
3. **Test authentication**: `echo "test" | gemini`
4. **Check workflow logs** in GitHub Actions tab
5. **Ensure repository permissions** for PR comments (Settings → Actions → Workflow permissions)
6. **Verify MCP server** is accessible if using clear history feature (port 8005)
7. **Check rate limits** - free tier has 60 requests/minute
8. **Test model availability**: `echo "test" | gemini -m gemini-2.5-pro`

### Common Issues

- **"Command not found"**: Gemini CLI not installed or not in PATH
  - Solution: Check `npm list -g @google/gemini-cli`
- **Authentication errors**: Run `gemini` directly to re-authenticate
  - Solution: Clear credentials with `gemini logout` then `gemini`
- **Rate limit exceeded**: Wait a few minutes and retry
  - Solution: Use gemini-2.5-flash for higher limits
- **No review posted**: Check if PR has proper permissions
  - Solution: Enable "Read and write permissions" in repo settings
- **Model not available**: Fallback to default model
  - Solution: Update CLI with `npm update -g @google/gemini-cli`

## Privacy Note

- Only the code diff and PR metadata are sent to Gemini
- No code is stored permanently by the AI service
- Reviews are supplementary to human code review
- Gemini follows Google's AI principles and data handling policies
- Consider using self-hosted alternatives for sensitive code

## References

- [Gemini CLI Documentation](https://github.com/google/gemini-cli)
- [Setup Guide](https://gist.github.com/AndrewAltimit/fc5ba068b73e7002cbe4e9721cebb0f5)
- [Gemini API Documentation](https://ai.google.dev/)
- [Model Comparison](https://ai.google.dev/models/gemini)

## Related Documentation

- [AI_AGENTS.md](AI_AGENTS.md) - Overview of AI agent collaboration
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - Context provided to Gemini
- [CLAUDE.md](CLAUDE.md) - Claude Code integration
