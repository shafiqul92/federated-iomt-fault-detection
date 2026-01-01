# GitHub Upload Guide

Complete step-by-step instructions for uploading your project to GitHub.

## Prerequisites

1. **Git installed** on your computer
   - Check: `git --version`
   - Download: https://git-scm.com/downloads

2. **GitHub account** created
   - Sign up at: https://github.com

## Step-by-Step Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com and sign in
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in the details:
   - **Repository name**: `fault-detection-iomt` (or your preferred name)
   - **Description**: "Fault Detection in Air Handling Units (AHU) using Federated Learning"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 2: Initialize Git Locally

Open terminal/command prompt in your project directory:

```bash
# Navigate to your project (if not already there)
cd C:\Users\shafiqul\Downloads\fault-detection-iomt

# Initialize git repository
git init

# Configure your git (if not done before)
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 3: Check .gitignore

Verify that `.gitignore` is present and working:

```bash
# View .gitignore contents
cat .gitignore

# Check what will be ignored
git status
```

Files that should be ignored (not uploaded):
- `*.pth` - Model checkpoints
- `*.csv` - Data files
- `plots/` - Generated plots
- `processed_splits_advanced/` - Processed data
- `venv/` - Virtual environment
- `__pycache__/` - Python cache

### Step 4: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what will be committed
git status
```

You should see files like:
- ✅ `README.md`
- ✅ `requirements.txt`
- ✅ `models/`, `data/`, `training/`, `evaluation/`, `config/`, `utils/`
- ✅ `.gitignore`
- ✅ Documentation files

You should NOT see:
- ❌ `*.pth` files
- ❌ `*.csv` files (except maybe in .gitignore exceptions)
- ❌ `venv/` directory
- ❌ Generated plots

### Step 5: Make Initial Commit

```bash
# Create initial commit
git commit -m "Initial commit: AHU Fault Detection System

- Centralized, FedAvg, and Edge-Aware FL implementations
- Hybrid LSTM+CNN+Attention model
- Real-time streaming evaluation
- Complete documentation"
```

### Step 6: Connect to GitHub Remote

After creating the repository on GitHub, copy the repository URL (it will look like):
- HTTPS: `https://github.com/yourusername/fault-detection-iomt.git`
- SSH: `git@github.com:yourusername/fault-detection-iomt.git`

Then run:

```bash
# Add remote repository (use your actual URL)
git remote add origin https://github.com/yourusername/fault-detection-iomt.git

# Verify remote was added
git remote -v
```

### Step 7: Push to GitHub

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

You may be prompted for:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your password)
  - Generate token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Select scopes: `repo` (full control)

### Step 8: Verify Upload

1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your files!

## Alternative: Using GitHub Desktop

If you prefer a graphical interface:

1. Download GitHub Desktop: https://desktop.github.com/
2. Sign in with your GitHub account
3. File → Add Local Repository
4. Select your project folder
5. Click "Publish repository" button
6. Choose name and visibility
7. Click "Publish Repository"

## Adding Data Files (Optional)

If you want to include example/sample data:

```bash
# Option 1: Add specific files
git add -f raw_data.csv  # -f forces adding even if in .gitignore

# Option 2: Create a .gitignore exception
# Add to .gitignore:
# !sample_data/raw_data_sample.csv

# Commit
git commit -m "Add sample data file"
git push
```

**Note**: Be careful with large data files. GitHub has file size limits (100MB warning, 1GB hard limit).

## Future Updates

After initial upload, to update your repository:

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Troubleshooting

### Issue: "Repository not found"
- Check the repository URL is correct
- Verify you have access permissions

### Issue: "Authentication failed"
- Use Personal Access Token instead of password
- Or set up SSH keys

### Issue: "File too large"
- Large files (>100MB) need Git LFS
- Or exclude them via .gitignore

### Issue: "Branch name mismatch"
```bash
# If main branch doesn't exist
git branch -M main
git push -u origin main

# If using master instead
git branch -M master
git push -u origin master
```

## Recommended Repository Settings

After uploading:

1. **Add Topics/Tags**: 
   - Go to repository → Topics
   - Add: `machine-learning`, `federated-learning`, `pytorch`, `iot`, `fault-detection`

2. **Add Description**: 
   - Edit repository description on main page

3. **Pin Important Repositories**:
   - Go to your profile → Customize pins
   - Pin this repository

4. **Create Releases** (optional):
   - Go to Releases → Create a new release
   - Tag version: v1.0.0
   - Title: "Initial Release"

## Quick Command Reference

```bash
# Initialize
git init

# Add files
git add .

# Commit
git commit -m "Your message"

# Connect to GitHub
git remote add origin <repository-url>

# Push
git push -u origin main

# Check status
git status

# View history
git log
```

## Security Notes

⚠️ **Important**:
- Never commit sensitive data (API keys, passwords)
- Review `.gitignore` before committing
- Use environment variables for secrets
- Consider making repository private if it contains proprietary data

## Success Checklist

- [ ] Git initialized
- [ ] All files added (respecting .gitignore)
- [ ] Initial commit made
- [ ] Remote repository connected
- [ ] Code pushed to GitHub
- [ ] Repository visible on GitHub
- [ ] README displays correctly
- [ ] All documentation files present

---

**Once uploaded, share your repository link! 🚀**

