---
layout: docs
title: Git Fundamentals
section: technology
---

# Git Fundamentals

## Overview

Git is a distributed version control system designed for non-linear development workflows and efficient handling of large projects. Its architecture enables multiple developers to work on the same codebase simultaneously while maintaining a complete history of all modifications. Git serves as the foundation for platforms including GitHub, GitLab, and Bitbucket.

## Core Concepts

### Repository
A Git repository is a database containing all project files and the complete revision history. Repositories can be local (on a developer's machine) or remote (on a server).

### Working Directory
The working directory contains the actual files that developers modify. It represents the current state of the project on the local filesystem.

### Staging Area (Index)
The staging area is an intermediate layer between the working directory and repository. Files must be explicitly added to the staging area before being committed to the repository.

### Commit
A commit represents a snapshot of the project at a specific point in time. Each commit contains:
- A unique SHA-1 hash identifier
- Author information
- Timestamp
- Commit message
- Pointer to parent commit(s)
- Reference to a tree object representing the project state

## Basic Commands

### Repository Initialization
```bash
git init                    # Initialize a new repository
git clone <repository-url>  # Clone an existing repository
```

### File Operations
```bash
git status                  # Display working directory status
git add <file>             # Stage specific file
git add .                  # Stage all changes
git commit -m "<message>"  # Create a commit with staged changes
```

### History and Inspection
```bash
git log                    # View commit history
git log --oneline         # Condensed commit history
git diff                  # Show unstaged changes
git diff --staged         # Show staged changes
```

### Remote Operations
```bash
git remote add origin <url>  # Add remote repository
git push origin <branch>     # Push commits to remote
git pull origin <branch>     # Fetch and merge remote changes
git fetch origin            # Fetch remote changes without merging
```

## Workflow Patterns

### Standard Git Operations Sequence
The Git version control workflow consists of four primary operations:
- **Modification**: Changes to working directory files
- **Staging**: Addition of changes to the index via `git add`
- **Committing**: Creation of immutable snapshots via `git commit`
- **Synchronization**: Remote repository updates via `git push`

### Feature Branch Operations
- **Branch Creation**: `git checkout -b feature-name`
- **Change Management**: Modifications followed by staging and committing
- **Remote Synchronization**: `git push origin feature-name`
- **Integration**: Pull request submission and merge after review

### Git Flow
A branching model designed around project releases:
- **main/master**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Individual feature branches
- **release/**: Preparation for production release
- **hotfix/**: Emergency fixes for production

### GitHub Flow
A workflow optimized for continuous deployment environments:
1. Create branch from main
2. Add commits
3. Open pull request
4. Discuss and review
5. Deploy for testing
6. Merge to main

## File States

Git tracks files in three states:

**Modified**: File has been changed but not staged
**Staged**: File marked for inclusion in next commit  
**Committed**: File safely stored in local repository

## Configuration

### User Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```

### Editor Configuration
```bash
git config --global core.editor "vim"
```

### View Configuration
```bash
git config --list          # Show all settings
git config user.name       # Show specific setting
```

## Common Operations

### Undoing Changes
```bash
git checkout -- <file>     # Discard working directory changes
git reset HEAD <file>      # Unstage file
git reset --soft HEAD~1    # Undo last commit, keep changes
git reset --hard HEAD~1    # Undo last commit, discard changes
```

### Branching
```bash
git branch                 # List branches
git branch <name>         # Create branch
git checkout <branch>     # Switch branches
git checkout -b <branch>  # Create and switch branch
git merge <branch>        # Merge branch into current
git branch -d <branch>    # Delete branch
```

### Stashing
```bash
git stash                 # Save uncommitted changes
git stash pop            # Apply and remove latest stash
git stash list           # List all stashes
git stash apply          # Apply stash without removing
```

## Standards and Conventions

### Commit Message Specification
- **Mood**: Imperative ("Add feature" not "Added feature")
- **Subject Line**: Maximum 50 characters
- **Format**: Blank line between subject and body
- **Body**: Maximum 72 characters per line, documenting rationale

#### Conventional Commits (2023-2024 Standard)
Many projects now follow the Conventional Commits specification:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code refactoring
- **test**: Test additions or modifications
- **chore**: Maintenance tasks

Example:
```
feat(auth): add OAuth2 integration

Implemented Google and GitHub OAuth2 providers with
proper token refresh and error handling.

Closes #123
```

### Commit Granularity
- **Atomic Commits**: Single logical change per commit
- **Build Integrity**: All commits must maintain build stability
- **Documentation**: Each commit requires descriptive message

### Branch Naming Convention
- **Format**: `<type>/<description>`
- **Standard Prefixes**: feature/, bugfix/, hotfix/, release/
- **Separator**: Hyphen for word separation

## .gitignore

The `.gitignore` file specifies which files Git should ignore:

```gitignore
# Compiled files
*.class
*.o
*.pyc

# Directories
build/
node_modules/
.venv/

# IDE files
.idea/
.vscode/
*.swp

# OS files
.DS_Store
Thumbs.db

# Environment files
.env
.env.local
```

## Advanced Git Features (2023-2024 Updates)

### Git Worktree
Manage multiple working trees attached to the same repository:
```bash
# Create a new worktree for a hotfix
git worktree add ../hotfix-branch hotfix/critical-bug

# List all worktrees
git worktree list

# Remove worktree
git worktree remove ../hotfix-branch
```

### Partial Clone and Sparse Checkout
Work with large repositories more efficiently:
```bash
# Clone with limited history
git clone --filter=blob:none --sparse <url>

# Enable sparse checkout
git sparse-checkout init --cone

# Add directories to checkout
git sparse-checkout set src/frontend docs
```

### Git Maintenance (Git 2.31+)
Automatic repository optimization:
```bash
# Enable automatic maintenance
git maintenance start

# Run maintenance tasks manually
git maintenance run --auto
```

## Security Best Practices (2024)

### Signing Commits
Ensure commit authenticity with GPG or SSH signatures:
```bash
# Configure GPG signing
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true

# Sign a specific commit
git commit -S -m "Signed commit"

# Verify signatures
git log --show-signature
```

### SSH Key Authentication (Recommended 2024)
GitHub deprecated password authentication. Use SSH keys:
```bash
# Generate ED25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## Git with AI Tools (2023-2024)

### GitHub Copilot Integration
- AI-powered code suggestions in your editor
- Context-aware commit message generation
- Automated PR descriptions

### GitLab AI Features
- Code suggestions
- Vulnerability explanation
- Code review summaries

## Performance Tips

### Large File Storage (LFS)
Handle large binary files efficiently:
```bash
# Track large files
git lfs track "*.psd"
git lfs track "*.zip"

# View tracked patterns
git lfs track

# Clone with LFS files
git lfs clone <repository>
```

### Optimizing Repository Performance
```bash
# Optimize repository
git gc --aggressive --prune=now

# Repack objects
git repack -a -d --depth=250 --window=250

# Clean unnecessary files
git clean -fd
```

## Integration with Modern Development

### CI/CD Integration
Git hooks for automated workflows:
```bash
# Pre-push hook example
#!/bin/sh
# .git/hooks/pre-push
npm test && npm run lint
```

### Monorepo Management
Tools for managing large codebases:
- **Nx**: Powerful monorepo build system
- **Lerna**: JavaScript monorepo tool
- **Bazel**: Google's build tool
- **Rush**: Microsoft's monorepo manager

## Error Recovery Procedures

### Commit Recovery
```bash
# Find lost commits
git reflog

# Restore lost commit
git checkout -b recovery-branch <commit-hash>
```

### Fixing Commit Mistakes
```bash
# Change last commit message
git commit --amend -m "New message"

# Add forgotten files to last commit
git add forgotten-file.txt
git commit --amend --no-edit
```

### Resolving Merge Conflicts
```bash
# Use mergetool
git mergetool

# Accept theirs/ours for specific files
git checkout --theirs path/to/file
git checkout --ours path/to/file
```

## Related Topics

- [Git Command Reference](git-reference.html) - Comprehensive command guide
- [Branching Strategies](branching.html) - Advanced branching workflows
- [CI/CD](ci-cd.html) - Continuous integration with Git
- [GitHub Actions](../devops/github-actions.html) - Automation workflows
- [GitLab CI](../devops/gitlab-ci.html) - GitLab's CI/CD platform

## References

- [Official Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book) - Free comprehensive guide
- [Git Reference Manual](https://git-scm.com/docs)
- [GitHub Skills](https://skills.github.com/) - Git practice environment
- [Conventional Commits](https://www.conventionalcommits.org/) - Commit message standard
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) - Original Git Flow model