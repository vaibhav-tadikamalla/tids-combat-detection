# 🚀 GitHub Setup Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - Repository name: `tids-combat-detection` or `military_project`
   - Description: "AI-powered combat impact detection system with 95.46% accuracy"
   - Visibility: **Public** (so others can access) or **Private**
   - ❌ DON'T check "Add README" (we already have one)
3. Click **Create repository**

---

## Step 2: Initialize Git Locally

Open PowerShell in your project folder and run:

```powershell
cd 'C:\Users\Vaibhav Tadikamalla\Desktop\military_project'

# Initialize git
git init

# Add all files (respects .gitignore)
git add .

# Make first commit
git commit -m "Initial commit: TIDS combat impact detection system"
```

---

## Step 3: Connect to GitHub

Replace `YOUR_USERNAME` with your actual GitHub username:

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/tids-combat-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: You'll be prompted for GitHub credentials. Use:
- Username: Your GitHub username
- Password: **Personal Access Token** (not your account password)

### How to Create Personal Access Token:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Select scopes: `repo` (all)
3. Copy the token and use it as password

---

## Step 4: Upload Model Files (3 Options)

### Option A: GitHub Releases (Easiest for Small Files <100MB)

1. Go to your repo on GitHub
2. Click **Releases** → **Create a new release**
3. Tag version: `v1.0.0`
4. Release title: "Guardian-Shield Model v1.0 (95.46% Accuracy)"
5. Drag and drop:
   - `impact_classifier.tflite`
   - `norm.npz`
6. Click **Publish release**

**Share link**: `https://github.com/YOUR_USERNAME/tids-combat-detection/releases/tag/v1.0.0`

---

### Option B: Hugging Face (Best for ML Models)

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model
huggingface-cli upload YOUR_USERNAME/guardian-shield impact_classifier.tflite
huggingface-cli upload YOUR_USERNAME/guardian-shield norm.npz
```

**Share link**: `https://huggingface.co/YOUR_USERNAME/guardian-shield`

---

### Option C: Google Drive (Quick & Easy)

1. Upload files to Google Drive
2. Right-click → Get link → **Anyone with the link**
3. Share the link in README.md

---

## Step 5: Update README with Download Links

Edit `README.md` and replace placeholder links:

```markdown
📥 **Download from**: 
- [GitHub Releases](https://github.com/YOUR_USERNAME/tids-combat-detection/releases/tag/v1.0.0)
- [Hugging Face](https://huggingface.co/YOUR_USERNAME/guardian-shield)
```

Then commit and push:

```powershell
git add README.md
git commit -m "Add model download links"
git push
```

---

## ✅ Done! Your Project is Now Public

**Share your repository**: `https://github.com/YOUR_USERNAME/tids-combat-detection`

Others can now:
- ✅ Clone your code
- ✅ Download your trained model
- ✅ Reproduce your results
- ✅ Contribute improvements

---

## 🔒 If You Made Repo Private

To give specific people access:
1. Repository → Settings → Collaborators
2. Add people → Enter their GitHub username
3. They'll get an invitation email

---

## 📱 Next Steps

1. Add a LICENSE file (MIT recommended)
2. Create GitHub Actions for CI/CD
3. Add badges to README (build status, coverage)
4. Enable GitHub Pages for project website
5. Star the repo and share on LinkedIn/Twitter! 🎉
