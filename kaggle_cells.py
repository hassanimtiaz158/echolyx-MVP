# ============================================================
# CELL 6: Push Results to GitHub
# ============================================================
%cd /kaggle/working/echolyx-MVP

import os
os.environ["GIT_TERMINAL_PROMPT"] = "0"

!git config user.email "kaggle@echolyx.ai"
!git config user.name "Kaggle Bot"
!git add -A

# Check what changed
status = !git status --porcelain
if status:
    print("Changed files:")
    !git status --short
    !git commit -m "chore: update training results from Kaggle run"
    
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "<YOUR_GITHUB_TOKEN>")
    !git remote set-url origin https://{GITHUB_TOKEN}@github.com/<YOUR_USERNAME>/echolyx-MVP.git
    !git push origin main
    print("Pushed to GitHub!")
else:
    print("No changes to push.")
