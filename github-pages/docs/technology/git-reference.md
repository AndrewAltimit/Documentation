---
layout: docs
title: "Git Command Reference"
---

# Git Command Reference

Git is a distributed version control system designed to handle everything from small to very large projects with speed and efficiency. This comprehensive reference covers essential Git commands, workflows, and the latest features introduced in 2023-2024.

## Repository Initialization

### Creating a New Repository
```bash
git init
```
Initializes a new Git repository in the current directory, creating a `.git` subdirectory containing all repository metadata.

### Cloning an Existing Repository
```bash
git clone <repository-url>
git clone <repository-url> <directory-name>
```
Creates a local copy of a remote repository. The second form allows specifying a custom directory name.

## Configuration

### User Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### View Configuration
```bash
git config --list
git config user.name
```

## Basic Workflow Commands

### Status and Differences
```bash
git status              # Show working tree status
git diff                # Show unstaged changes
git diff --staged       # Show staged changes
git diff HEAD           # Show all changes since last commit
```

### Staging Changes
```bash
git add <file>          # Stage specific file
git add .               # Stage all changes in current directory
git add -A              # Stage all changes in repository
git add -p              # Interactive staging
```

### Committing Changes
```bash
git commit -m "Commit message"       # Commit with inline message
git commit                           # Opens editor for message
git commit -am "Message"             # Stage and commit tracked files
git commit --amend                   # Modify last commit
```

## History and Inspection

### Viewing History
```bash
git log                              # Show commit history
git log --oneline                    # Compact format
git log --graph                      # ASCII graph of branches
git log --stat                       # Include file changes
git log -p                           # Show patches
git log --author="Name"              # Filter by author
git log --since="2 weeks ago"        # Time-based filtering
```

### Examining Commits
```bash
git show                             # Show last commit
git show <commit-hash>               # Show specific commit
git show <commit-hash>:<file>        # Show file at specific commit
```

## Branching and Merging

### Branch Management
```bash
git branch                           # List local branches
git branch -a                        # List all branches
git branch <branch-name>             # Create new branch
git branch -d <branch-name>          # Delete branch (safe)
git branch -D <branch-name>          # Force delete branch
```

### Switching Branches
```bash
git checkout <branch-name>           # Switch to branch
git checkout -b <branch-name>        # Create and switch to branch
git switch <branch-name>             # Modern alternative to checkout (Git 2.23+)
git switch -c <branch-name>          # Create and switch (modern)
git switch -                         # Switch to previous branch
```

### Merging
```bash
git merge <branch-name>              # Merge branch into current
git merge --no-ff <branch-name>      # Force merge commit
git merge --abort                    # Abort conflicted merge
```

## Remote Operations

### Remote Management
```bash
git remote                           # List remotes
git remote -v                        # Show remote URLs
git remote add <name> <url>          # Add new remote
git remote remove <name>             # Remove remote
git remote rename <old> <new>        # Rename remote
```

### Synchronization
```bash
git fetch                            # Download remote changes
git fetch <remote>                   # Fetch from specific remote
git pull                             # Fetch and merge
git pull --rebase                    # Fetch and rebase
git push                             # Upload local changes
git push <remote> <branch>           # Push specific branch
git push -u origin <branch>          # Set upstream and push
```

## Undoing Changes

### Working Directory
```bash
git restore <file>                   # Discard changes (Git 2.23+)
git checkout -- <file>               # Discard changes (legacy)
git clean -fd                        # Remove untracked files
```

### Staging Area
```bash
git restore --staged <file>          # Unstage file (Git 2.23+)
git reset HEAD <file>                # Unstage file (legacy)
```

### Commits
```bash
git reset --soft HEAD~1              # Undo commit, keep changes staged
git reset --mixed HEAD~1             # Undo commit, unstage changes (default)
git reset --hard HEAD~1              # Undo commit, discard changes
git revert <commit>                  # Create new commit undoing changes
git revert --no-commit <commit>      # Revert without committing
git revert -m 1 <merge-commit>       # Revert a merge commit
```

## Stashing

```bash
git stash                            # Save changes temporarily
git stash push -m "Description"      # Stash with message
git stash list                       # List stashes
git stash apply                      # Apply most recent stash
git stash apply stash@{n}            # Apply specific stash
git stash pop                        # Apply and remove stash
git stash drop stash@{n}             # Remove specific stash
git stash clear                      # Remove all stashes
```

## Tags

```bash
git tag                              # List tags
git tag <tag-name>                   # Create lightweight tag
git tag -a <tag-name> -m "Message"   # Create annotated tag
git tag -d <tag-name>                # Delete local tag
git push origin <tag-name>           # Push tag to remote
git push origin --tags               # Push all tags
```

## Advanced Operations

### Rebasing
```bash
git rebase <branch>                  # Rebase current branch
git rebase -i HEAD~n                 # Interactive rebase last n commits
git rebase --continue                # Continue after resolving conflicts
git rebase --abort                   # Cancel rebase
```

### Cherry-picking
```bash
git cherry-pick <commit>             # Apply specific commit
git cherry-pick --continue           # Continue after resolving conflicts
git cherry-pick --abort              # Cancel cherry-pick
```

### Searching
```bash
git grep <pattern>                   # Search in working directory
git log -S <string>                  # Find commits that add/remove string
git log --grep=<pattern>             # Search commit messages
```

## Performance and Maintenance

### Optimization
```bash
git gc                               # Garbage collection
git prune                            # Remove unreachable objects
git fsck                             # Check repository integrity
```

### Large Files
```bash
git lfs track "*.psd"                # Track file type with Git LFS
git lfs ls-files                     # List LFS tracked files
```

## Common Workflows

### Feature Branch Workflow
```bash
git checkout -b feature/new-feature  # Create feature branch
# Make changes
git add .
git commit -m "Add new feature"
git push -u origin feature/new-feature
# Create pull request
```

### Hotfix Workflow
```bash
git checkout main
git checkout -b hotfix/critical-fix
# Make fix
git add .
git commit -m "Fix critical issue"
git checkout main
git merge hotfix/critical-fix
git push
```

## Best Practices

1. **Commit Messages**: Use clear, descriptive messages in imperative mood
2. **Atomic Commits**: Each commit should represent one logical change
3. **Branch Naming**: Use descriptive names (feature/, bugfix/, hotfix/)
4. **Regular Commits**: Commit frequently but meaningfully
5. **Pull Before Push**: Always sync with remote before pushing
6. **Review Before Commit**: Use `git diff --staged` before committing

## Advanced Features (2023-2024)

### Worktrees
```bash
git worktree add <path> <branch>     # Create new worktree
git worktree list                    # List all worktrees
git worktree remove <path>           # Remove worktree
git worktree prune                   # Clean up stale worktrees
```

### Sparse Checkout
```bash
git sparse-checkout init             # Initialize sparse checkout
git sparse-checkout set <dir1> <dir2> # Set directories to include
git sparse-checkout list             # Show current sparse patterns
git sparse-checkout disable          # Disable sparse checkout
```

### Maintenance and Performance
```bash
git maintenance start                # Enable automatic maintenance
git maintenance run                  # Run maintenance tasks
git maintenance stop                 # Disable automatic maintenance
git commit-graph write               # Generate commit graph
git multi-pack-index write           # Optimize pack files
```

### Advanced Diff and Merge
```bash
git diff --color-words               # Show word-level differences
git diff --word-diff                 # Alternative word diff format
git diff --name-status               # Show only file names and status
git diff --check                     # Check for whitespace errors
git merge --no-ff                    # Force merge commit
git merge --squash                   # Squash branch into single commit
git rerere                           # Reuse recorded resolution
```

### Bundle Operations
```bash
git bundle create <file> <refs>      # Create bundle file
git bundle verify <file>             # Verify bundle
git bundle list-heads <file>         # List references in bundle
git clone <bundle-file> <dir>        # Clone from bundle
```

### Bisect for Bug Finding
```bash
git bisect start                     # Start bisect session
git bisect bad                       # Mark current commit as bad
git bisect good <commit>             # Mark known good commit
git bisect skip                      # Skip current commit
git bisect reset                     # End bisect session
git bisect run <script>              # Automate bisect with script
```

### Signing and Verification
```bash
git config --global user.signingkey <key-id>  # Set GPG key
git config --global commit.gpgsign true        # Auto-sign commits
git config --global tag.gpgsign true           # Auto-sign tags
git commit -S -m "Signed commit"               # Sign specific commit
git tag -s <tag-name> -m "Signed tag"          # Create signed tag
git verify-commit <commit>                      # Verify commit signature
git verify-tag <tag>                            # Verify tag signature
```

### SSH Signing (Git 2.34+)
```bash
git config gpg.format ssh                       # Use SSH for signing
git config user.signingkey ~/.ssh/id_ed25519   # Set SSH key
git config gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

## Troubleshooting

### Common Issues

**Detached HEAD State**
```bash
git checkout <branch-name>           # Return to branch
git checkout -b <new-branch>         # Create branch from current state
git switch -c <new-branch>           # Modern way to create branch
```

**Merge Conflicts**
```bash
# Edit conflicted files manually
git add <resolved-files>
git commit                           # Complete merge
# Or use merge tool
git mergetool                        # Launch configured merge tool
```

**Wrong Branch Commits**
```bash
git cherry-pick <commit>             # Copy commit to correct branch
git reset --hard HEAD~1              # Remove from wrong branch
# Or move last n commits to new branch
git branch <new-branch>
git reset --hard HEAD~n
git checkout <new-branch>
```

**Large File Issues**
```bash
git filter-branch --tree-filter 'rm -f <large-file>' HEAD
# Or use modern alternative
git filter-repo --path <large-file> --invert-paths
```

**Corrupted Repository**
```bash
git fsck --full                      # Check repository integrity
git gc --prune=now                   # Clean up repository
git reflog expire --expire=now --all # Expire all reflog entries
```

## Performance Optimization

### Config Optimizations
```bash
git config core.preloadindex true    # Speed up git status
git config core.fscache true         # Enable file system cache (Windows)
git config core.untrackedCache true  # Cache untracked files
git config feature.manyFiles true    # Optimize for many files
```

### Large Repository Handling
```bash
# Shallow clone
git clone --depth 1 <url>            # Clone only latest commit
git clone --shallow-since=<date>     # Clone from specific date
git clone --shallow-exclude=<rev>    # Exclude specific revision

# Partial clone
git clone --filter=blob:none <url>   # Clone without file contents
git clone --filter=tree:0 <url>      # Clone without trees
```

## Integration with Development Tools

### VS Code Integration
```bash
git config --global core.editor "code --wait"
git config --global diff.tool vscode
git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
```

### GitHub CLI Integration
```bash
gh repo clone <owner>/<repo>         # Clone with GitHub CLI
gh pr create                         # Create pull request
gh pr checkout <number>              # Checkout PR locally
gh issue create                      # Create issue
```

### Pre-commit Hooks
```bash
# Install pre-commit framework
pip install pre-commit
pre-commit install                   # Install git hooks
pre-commit run --all-files          # Run on all files
```

## Security Best Practices

### Removing Sensitive Data
```bash
# Remove file from all commits
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch <file>' \
  --prune-empty --tag-name-filter cat -- --all

# Modern alternative using git-filter-repo
git filter-repo --path <sensitive-file> --invert-paths
```

### Secret Scanning
```bash
# Use tools like git-secrets
git secrets --install
git secrets --register-aws          # Register AWS patterns
git secrets --scan                  # Scan for secrets
```

## Related Git Documentation

- [Git Version Control](git.html) - Deep dive into Git internals and architecture
- [Git Crash Course](git-crash-course.html) - Beginner-friendly introduction
- [Branching Strategies](branching.html) - Git Flow, GitHub Flow, and team workflows
- [CI/CD Pipelines](ci-cd.html) - Continuous integration with Git

## Additional Resources

- [Pro Git Book](https://git-scm.com/book) - Comprehensive Git guide
- [Git Documentation](https://git-scm.com/docs) - Official reference
- [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain) - Deep dive
- [GitHub Skills](https://skills.github.com/) - Interactive tutorials
- [Atlassian Git Tutorial](https://www.atlassian.com/git) - Visual guides
- [Git Flight Rules](https://github.com/k88hudson/git-flight-rules) - What to do when things go wrong

---

## See Also
- [Git Version Control](git.html) - Git architecture, internals, and distributed version control theory
- [Branching Strategies](branching.html) - Workflow patterns including Git Flow, GitHub Flow, and trunk-based development
- [CI/CD](ci-cd.html) - Automating Git workflows with continuous integration and deployment pipelines