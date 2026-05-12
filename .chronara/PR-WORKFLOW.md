# Chronara — Upstream PR Submission Workflow

This file lives on Chronara's `CG-8663/cuda-oxide` fork to document our internal contribution workflow. Anyone submitting an upstream PR from this fork must follow this sequence.

## The non-negotiable rule

> **James reviews the rendered PR body on GitHub in full before any upstream PR posts.** No exceptions.

A draft PR that gets converted to ready-for-review counts as a posting; the review must happen before the draft is created, OR the upstream PR must use `--draft` so James can see the rendered state on GitHub itself before flipping it ready.

## Approved sequence

1. **Branch on the fork.**
   ```bash
   git clone git@github.com:CG-8663/cuda-oxide.git /tmp/cuda-oxide-pr/cuda-oxide
   cd /tmp/cuda-oxide-pr/cuda-oxide
   git remote add upstream https://github.com/NVlabs/cuda-oxide.git
   git fetch upstream main
   git checkout -b <branch-name> upstream/main
   ```

2. **Stage changes.** Make file changes, get them right locally. Run any lint / build verification on `gx10-001` if relevant. **Use Chronara identity only — no agent attribution (no `Co-Authored-By: <persona>`), no AI-tool attribution (no `Generated with ...` footer).** Single author: `Chronara Group <230511811+CG-8663@users.noreply.github.com>`.

3. **Commit + push to fork.**
   ```bash
   git config user.email "230511811+CG-8663@users.noreply.github.com"
   git config user.name "Chronara Group"
   git add <files>
   git commit -F <commit-msg-file>
   git push -u origin <branch-name>
   ```
   Fork pushes are reversible — force-pushing to overwrite is fine pre-submission.

4. **Write the PR body to a file.** `/tmp/cuda-oxide-pr/PR_BODY.md` is the convention. The body must be inspectable as a file before it goes near `gh pr create`.

5. **🔴 REVIEW GATE — James reads the full body on GitHub before any submission to upstream.**

   Two ways to surface the rendered state for review:

   **Option A — Use the fork's `pull/new` URL.** This opens GitHub's PR-create UI but does NOT submit until "Create pull request" is clicked. James pastes the body content from `PR_BODY.md`, sees the rendered preview (tables, links, markdown all rendered), and either clicks Create or doesn't. Most controlled.

   ```
   https://github.com/CG-8663/cuda-oxide/pull/new/<branch-name>
   ```

   **Option B — `gh pr create --draft`.** Opens the PR on upstream as DRAFT. Visible to upstream but explicitly not-ready-for-review. James reviews the rendered state on the actual upstream PR page, then runs `gh pr ready` (or clicks "Ready for review") to flip it. This option means the draft is briefly upstream-visible — fine for contribution workflow, but choose option A for sensitive content.

   Whichever option, **the review is on the rendered GitHub state, not on a chat preview**.

6. **Submit.** Either click the GitHub UI button (option A) or `gh pr ready` (option B).

## Anti-pattern (what NOT to do)

Do not run `gh pr create --repo NVlabs/cuda-oxide ...` (non-draft, non-`pull/new`) directly from a shell session without the review gate. Even with prior verbal authorization, the review must happen on the rendered GitHub state. Verbal "yes go" in a Claude session is not the review gate — the GitHub-side review is.

## What goes in / what stays out (externally-visible artefacts)

**In:**
- `Chronara Group` / `Chronara AI` / `Chronara Dev Team` for org attribution.
- `James Tervit` for personally-attributed work (with a Chronara-identity email or GitHub noreply).
- Specific named team members where they're being explicitly introduced (e.g., a published blog post about the team).
- Substantive technical content.

**Out:**
- Agent persona names (Rin, Marcus, Tycho, Marek, etc.) in commit trailers / PR bodies / public docs.
- AI tool attribution (`Generated with [Claude Code]`, `Co-Authored-By: Claude`, etc.).
- Internal email addresses tied to agent personas (`kernel-smith@chronara.io`, etc.).

## When in doubt

Pause and ask James before posting. The cost of a 5-minute confirmation step is negligible compared to the cost of cleaning up a published artefact.
