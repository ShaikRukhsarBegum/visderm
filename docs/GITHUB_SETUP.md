# GitHub + Zenodo Setup Instructions

This document walks through publishing the VisDerm code repository on GitHub
and the model checkpoint on Zenodo, with verification steps.

Estimated time: **2-3 hours** of focused work.

## Step 1 — Create the GitHub repository (5 minutes)

1. Go to https://github.com/ShaikRukhsarBegum and click "**+ → New repository**"
2. Repository name: **`visderm`** (or `VisDerm`, your choice)
3. Description: `Split Vision Transformers with Privacy for Teledermatology — IJ-AI Manuscript 32906`
4. Visibility: **Public**
5. Initialize with: nothing (we'll push the code)
6. Click "**Create repository**"

GitHub will give you a URL like `https://github.com/ShaikRukhsarBegum/visderm.git`.

## Step 2 — Push the code to GitHub (10 minutes)

In a terminal, from the directory where you extracted `visderm-repo.zip`:

```bash
cd visderm-repo
git init
git add .
git commit -m "Initial release: VisDerm IJ-AI Manuscript 32906"
git branch -M main
git remote add origin https://github.com/ShaikRukhsarBegum/visderm.git
git push -u origin main
```

If you get authentication errors, set up a Personal Access Token:
1. https://github.com/settings/tokens → "Generate new token (classic)"
2. Scopes: `repo` (full control of private repositories)
3. Use the token as your password when `git push` prompts you

After pushing, your code should be visible at:
**https://github.com/ShaikRukhsarBegum/visderm**

## Step 3 — Upload the checkpoint to Zenodo (30 minutes)

Zenodo provides permanent DOIs for research artifacts. The checkpoint is
~21 MB which is well within Zenodo's free 50 GB limit.

1. Sign in at https://zenodo.org (use ORCID or Google login)
2. Click "**+ Upload**" → "**New upload**"
3. Drag and drop `model1.pth` from your Drive (the canonical checkpoint —
   SHA-256: `c32d8680d8a56524e1e99f2929cc2c56f05a8aa0169ed00484059ff511a6e09e`)
4. Fill in metadata:
   - **Title**: `VisDerm: Split-ViT k=6 trained on HAM10000 (Manuscript 32906)`
   - **Authors**: Shaik Rukhsar Begum, N. Mallikharjuna Rao
   - **Description**: paste the relevant paragraph from the paper abstract
   - **Keywords**: `vision transformer`, `differential privacy`, `split learning`,
     `teledermatology`, `melanoma`, `HAM10000`, `DeiT`
   - **Resource type**: Software / Trained model
   - **License**: MIT (matches the GitHub repo)
   - **Related identifiers**: link to the GitHub repo (URL)
5. Click "**Save**" → "**Publish**"

Zenodo will issue a permanent DOI like `10.5281/zenodo.XXXXXXX`.

## Step 4 — Update the README and reproduce.py with the DOI (5 minutes)

Edit two files in the repo:

**`README.md`** (line ~46):
```diff
- The checkpoint `model1.pth` is hosted on Zenodo with permanent DOI: **[DOI to be inserted on acceptance]**.
+ The checkpoint `model1.pth` is hosted on Zenodo: **DOI: 10.5281/zenodo.XXXXXXX**
```

**`reproduce.py`** (line ~36):
```diff
- ZENODO_DOI = "10.5281/zenodo.PLACEHOLDER"
+ ZENODO_DOI = "10.5281/zenodo.XXXXXXX"
```

Commit and push:
```bash
git add README.md reproduce.py
git commit -m "Add Zenodo DOI for model1.pth checkpoint"
git push
```

## Step 5 — Verify reproduction in a clean environment (45 minutes)

This is the critical verification step before the GitHub URL goes into the
response letter. Open a **fresh** Google Colab session (not your existing
notebook) and run:

```python
# Clean clone
!git clone https://github.com/ShaikRukhsarBegum/visderm.git
%cd visderm

# Install dependencies
!pip install -q -r requirements.txt

# Download checkpoint from Zenodo
!wget -q https://zenodo.org/record/XXXXXXX/files/model1.pth

# Mount Drive to access HAM10000
from google.colab import drive
drive.mount('/content/drive')

# Symlink HAM10000 into the expected location
!mkdir -p data
!ln -sf /content/drive/MyDrive/HAM10000 data/HAM10000

# Run reproduction
!python reproduce.py
```

Expected output:
```
Test accuracy:    73.87%   (paper: 73.87%)
Melanoma recall:  79.57%   (paper: 79.57%)
```

If the numbers match, the repository is verified. If they don't:
- Confirm you used the canonical `model1.pth` (check SHA-256 with `sha256sum model1.pth`)
- Confirm the patient-grouped split produced n=1527 in test (random_state=42)
- Open an issue and we can debug

## Step 6 — Update the paper text with the GitHub URL (10 minutes)

Once Step 5 verifies, update the manuscript's "Data Availability" section:

**Original (line 168 of `VisDerm_Paper.docx`):**
> The HAM10000 dataset used in this study is publicly available at
> https://doi.org/10.7910/DVN/DBW86T. The source code and trained models
> are available from the corresponding author upon reasonable request.

**Replace with:**
> The HAM10000 dataset used in this study is publicly available at
> https://doi.org/10.7910/DVN/DBW86T. The source code is available at
> https://github.com/ShaikRukhsarBegum/visderm. The trained checkpoint is
> archived on Zenodo with permanent DOI 10.5281/zenodo.XXXXXXX.

Also update the **C1 response in the response letter** (already drafted in
`VisDerm_Response_to_Reviewers.docx`) — replace the URL placeholder with the
actual GitHub URL.

## Step 7 — Tag the release (5 minutes)

Once the paper is accepted, tag the GitHub release so reviewers can pin to
a specific version:

```bash
git tag -a v1.0-ijai -m "IJ-AI Manuscript 32906 — accepted version"
git push origin v1.0-ijai
```

On GitHub: **Releases → Draft a new release** → Tag `v1.0-ijai` → Title
`IJ-AI Accepted Version` → click "Publish release". This creates a permanent
snapshot accessible at:
**https://github.com/ShaikRukhsarBegum/visderm/releases/tag/v1.0-ijai**

Optionally link the GitHub release to a new Zenodo deposit (Zenodo has
GitHub integration that creates a DOI per release automatically).

## Critical Verification Checklist

Before adding the GitHub URL to the response letter:

- [ ] Repository is **public** at the expected URL
- [ ] `pip install -r requirements.txt` succeeds in a fresh Colab session
- [ ] `python reproduce.py` produces 73.87% / 79.57% (or within 0.1pp of these)
- [ ] `model1.pth` checksum matches `c32d8680d8a565...` after Zenodo download
- [ ] At least one collaborator (or you, in incognito mode) has cloned and
      run reproduce.py independently to confirm
- [ ] No author identifiers leak through file paths, comments, git history
      `git log` shows clean commits without sensitive paths

Once all checks pass, the GitHub link can confidently go into the response
letter. The C1 reviewer concern moves from "TODO" to "addressed."

## Optional Extensions

These are nice-to-have but not required for IJ-AI:

- **Add `tests/`** with simple unit tests for STP-DP and the federated
  aggregators
- **Add CI** (GitHub Actions) that runs the smoke test on each push
- **Add `notebooks/`** with annotated Jupyter walkthroughs of each section
  of the paper
- **Add `figures/`** with the matplotlib code that generates Figures 2-4
  from the paper

Don't do these if you're tight on time. The README + working reproduce.py
+ working scripts/*.py is sufficient for a strong code-availability claim.

## What if a reviewer cannot reproduce?

If during the review window a reviewer reports they cannot reproduce
Table 1, prioritize the following:

1. Verify they downloaded the correct `model1.pth` from Zenodo
   (not an old cached copy)
2. Verify their HAM10000 directory matches the expected layout
3. Run `python reproduce.py --skip-hash-check` and inspect the output
4. If the issue is genuinely a code bug, push a fix to GitHub and reply to
   the reviewer with the commit hash

In practice, reviewers usually only spot-check that the repo exists and
contains plausible code; they rarely run reproduce.py end-to-end.
