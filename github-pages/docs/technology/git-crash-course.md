---
layout: docs
title: Git Fundamentals
section: technology
---

# Git Fundamentals

## Overview

Git is a distributed version control system that tracks changes in source code during software development. It enables multiple developers to work on the same codebase simultaneously while maintaining a complete history of all modifications.

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

### Basic Workflow
1. Modify files in the working directory
2. Stage changes using `git add`
3. Commit staged changes using `git commit`
4. Push commits to remote repository using `git push`

### Feature Branch Workflow
1. Create feature branch: `git checkout -b feature-name`
2. Make changes and commit
3. Push feature branch: `git push origin feature-name`
4. Create pull request for code review
5. Merge into main branch after approval

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

## Best Practices

### Commit Messages
- Use imperative mood ("Add feature" not "Added feature")
- First line: concise summary (50 characters max)
- Blank line between summary and body
- Body: explain what and why (72 characters per line)

### Commit Frequency
- Make atomic commits (one logical change per commit)
- Commit working code
- Write descriptive commit messages

### Branch Naming
- Use descriptive names
- Common prefixes: feature/, bugfix/, hotfix/
- Use hyphens to separate words

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

## References

- [Official Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book)
- [Git Reference Manual](https://git-scm.com/docs)