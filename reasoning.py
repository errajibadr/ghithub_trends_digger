```bash
BASE=61175a17ece5b99ed0e0cc24f3463931a29286af

git status --short
git cat-file -e "${BASE}^{commit}"
git bundle verify /path/to/project.bundle
git fetch /path/to/project.bundle refs/heads/main
git merge --ff-only FETCH_HEAD
```
