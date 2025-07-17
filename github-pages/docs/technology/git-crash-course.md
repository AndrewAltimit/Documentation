---
layout: docs
title: Git in 5 Minutes
difficulty_level: beginner
section: technology
---

# Git: Your Time Machine for Code (5 Minute Read)

{% include learning-breadcrumb.html 
   path=site.data.breadcrumbs.technology 
   current="Git in 5 Minutes"
   alternatives=site.data.alternatives.git_beginner 
%}

{% include skill-level-navigation.html 
   current_level="beginner"
   topic="Git"
   intermediate_link="/docs/technology/branching/"
   advanced_link="/docs/technology/git/"
%}

## What is Git? 

Imagine you're writing a book. Every time you make major changes, you save a new copy: "MyBook_v1.docx", "MyBook_v2_final.docx", "MyBook_v2_FINAL_FINAL.docx" (we've all been there).

**Git is like having a magical filing cabinet** that:
- Saves every version automatically
- Shows you exactly what changed between versions
- Lets multiple people work on the same book without chaos
- Can teleport you back to any previous version instantly

## Why Should You Care?

Without Git, coding is like:
- Walking a tightrope without a safety net
- Writing a novel with no way to undo mistakes
- Trying to collaborate via email attachments (nightmare!)

With Git, you can:
- **Experiment fearlessly** - Break things! You can always go back
- **Collaborate smoothly** - No more "whose version is newest?"
- **Track your progress** - See how your code evolved over time

## The Basic Workflow (Restaurant Kitchen Analogy)

Think of Git like a restaurant kitchen:

1. **Working Directory** = Your cutting board
   - Where you actively prepare (edit) your files
   
2. **Staging Area** = The pass (where finished dishes wait)
   - Where you place files that are ready to be saved
   
3. **Repository** = The recipe book
   - Where all your saved versions live permanently

### The Flow:
```
Edit files → Stage them → Commit (save) them
(Prep food) → (Ready to serve) → (Add to menu)
```

## Essential Commands (Your Git Toolbox)

### Setting Up
```bash
git init                    # Create a new recipe book
git clone [url]            # Copy someone else's recipe book
```

### Daily Workflow
```bash
git status                 # What's on my cutting board?
git add [file]            # Put this dish on the pass
git add .                 # Put ALL dishes on the pass
git commit -m "message"   # Save to the recipe book
```

### Collaboration
```bash
git push                  # Share your recipes with others
git pull                  # Get others' latest recipes
```

### Time Travel
```bash
git log                   # See all saved versions
git checkout [commit]     # Visit a past version
```

## Visual Guide: Your First Git Save

```
1. You edit hello.py (cooking)
   └─→ File is "modified" (red in git status)

2. git add hello.py (plating)
   └─→ File is "staged" (green in git status)

3. git commit -m "Add greeting feature"
   └─→ Version saved forever with your message

4. git push (if using GitHub/GitLab)
   └─→ Backed up online for safety
```

## Try This Now! (3 Minutes)

### Exercise 1: Your First Repository
```bash
# In your terminal:
mkdir my-first-repo
cd my-first-repo
git init
echo "Hello Git!" > readme.txt
git add readme.txt
git commit -m "My first commit!"
```

### Exercise 2: See What Git Sees
```bash
# Make a change:
echo "Git is awesome!" >> readme.txt

# Check status:
git status
# (See the red "modified" file?)

# Stage and commit:
git add readme.txt
git commit -m "Add enthusiasm"

# See your history:
git log --oneline
```

## Common "Aha!" Moments

- **"Commits are snapshots, not just saves"** - Each commit captures your entire project state
- **"Branches are parallel universes"** - Work on features without affecting the main code
- **"Git and GitHub are different"** - Git is the tool, GitHub is one place to store repositories

## What NOT to Do (Save Yourself Pain)

- ❌ Don't commit passwords or API keys
- ❌ Don't panic if you mess up (everything is fixable)
- ❌ Don't commit giant files (videos, datasets)
- ❌ Don't work directly on the main branch (create feature branches)

## Ready for More?

{% include difficulty-helper.html 
   current_level="beginner"
   harder_link="/docs/technology/branching/"
   prerequisites=site.data.prerequisites.git_beginner
   advanced_topics=site.data.advanced_topics.git
%}

You've learned the Git basics! When you're ready to level up:

- **[Branching & Collaboration →](/docs/technology/branching.html)** - Work with branches and teams (Intermediate)
- **[Git Internals →](/docs/technology/git.html)** - How Git really works under the hood (Advanced)
- **Practice Project**: Create a repository for your personal notes or a small project

{% include progressive-disclosure.html 
   sections=site.data.git_topics.beginner_progression
   initial_depth="overview"
%}

## Quick Reference Card

| Task | Command | Kitchen Analogy |
|------|---------|----------------|
| Start new project | `git init` | Open new restaurant |
| Save changes | `git add` + `git commit` | Prep + Add to menu |
| See status | `git status` | Check your stations |
| View history | `git log` | Browse recipe book |
| Share work | `git push` | Publish cookbook |

---

**Remember**: Everyone makes Git mistakes. The beauty is that almost everything is reversible. Start small, practice often, and soon Git will feel as natural as saving a file!