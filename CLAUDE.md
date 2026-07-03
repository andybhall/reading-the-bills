# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude Code Research Instructions

*Last updated: February 2026*

---

## Foundation

Before beginning any research task, review and internalize the principles in `SOUL.md`. That document governs your judgment and values; this one governs your execution.

When in doubt about a judgment call, refer to `SOUL.md`. When in doubt about how to do something, refer to this document.

---

## Environment and Tools

- **Always use `python3` and `pip3`**, never `python` or `pip`.
- **Default language is Stata** unless there is an affirmative reason to use something else (e.g., scraping, databases, tasks requiring Python libraries). This comes from SOUL.md.
- This home directory contains multiple independent research project directories. Each project may have its own CLAUDE.md, README, or SOUL.md—check before starting work.

## Standard Project Structure (from SOUL.md)

Projects follow this folder convention:
- **Draft** — TeX file and related materials
- **Original Data** — raw data from primary sources (never modify these)
- **Modified Data** — data files produced by code
- **Code**
- **Literature** — relevant research papers
- **Notes**

Results always pull from files that analysis code produces so that no results are ever mistranscribed or out of date. No statistical output is hard-coded into writing.

---

## Critical Workflow Requirements

**These rules are non-negotiable.**

1. **Never claim something works without running a test to prove it.** After writing any code, immediately write and run a test. If you cannot test it, say so explicitly. "It should work" is not acceptable—show me that it works.

2. **Work modularly.** Complete one module or task at a time. After each module, report what you built, show test results, and wait for confirmation before proceeding. Do not build an entire pipeline and then test it at the end.

3. **Iterate and fix errors yourself.** Do not rely on me to report errors back to you. Run the code, observe the output, and fix problems before presenting results. If you hit an error, debug it. If you can't fix it after genuine effort, then ask for help.

4. **Be explicit about unknowns.** If you're uncertain about something, say so. Don't guess. Don't confabulate. "I don't know" is an acceptable answer.

---

## Before Starting Any Task

- **Confirm you understand the goal.** Restate what you think I'm asking for. If there's ambiguity, ask before proceeding.

- **Check for existing work.** Look at what already exists in the project folder before creating new files. Don't duplicate work that's already been done.

- **Identify dependencies.** What data, files, or prior steps does this task depend on? Verify they exist and are in the expected state.

- **Plan before coding.** For any non-trivial task, outline your approach before writing code. This helps catch misunderstandings early.

---

## Data Collection and Preparation

- **Always verify completeness.** After collecting data, check that you have collected everything you intended to collect. Don't assume—verify. Count observations, check date ranges, confirm coverage.

- **Document sources.** For every piece of data you collect, record where it came from and when you accessed it.

- **Preserve raw data.** Never modify original data files. All transformations happen in code, producing new files in the Modified Data folder.

- **Check for anomalies.** After loading or transforming data, run basic sanity checks: observation counts, missing values, unexpected values, distributions of key variables.

- **Verify merges.** After any merge, check that the result makes sense. How many observations matched? How many didn't? Are there unexpected duplicates? Does the final N match what you expected?

---

## Code Standards

- **Use clear, descriptive names.** Variables, functions, and files should have names that make their purpose obvious. Avoid abbreviations that aren't universally understood.

- **Comment the why, not the what.** Comments should explain reasoning and intent, not describe what the code obviously does.

- **One task per script.** Each script should do one coherent thing. If a script is doing multiple unrelated tasks, break it up.

- **Paths should be relative.** Don't use absolute paths that only work on one machine. Use relative paths from the project root.

- **Handle errors gracefully.** Anticipate what could go wrong and handle it explicitly. Don't let scripts fail silently.

- **Print progress for long operations.** If something takes more than a few seconds, print status updates so we know it's working.

---

## Analysis

- **Start simple.** Run the simplest version of the analysis first. Add complexity only after the simple version works.

- **Verify each step.** Don't chain together multiple transformations and hope the output is right. Check intermediate results.

- **Sanity check results.** Do the results make sense? Are coefficients the right order of magnitude? Are signs plausible? Is the sample size what you expected?

- **Save output systematically.** Results should be saved to files that can be read by the paper's TeX file. No manual transcription of numbers.

- **Document specifications.** For every regression or analysis, it should be clear exactly what the specification is: outcome variable, treatment variable, controls, fixed effects, standard error clustering.

---

## Reporting and Communication

- **Lead with the bottom line.** When reporting results, start with what I most need to know. Details can follow.

- **Show, don't just tell.** When you say something worked, show me the output. When you say the data looks clean, show me the checks you ran.

- **Be specific about uncertainty.** Don't say "there might be issues." Say exactly what the issue is and how confident you are.

- **Flag decisions you made.** If you had to make a choice that could reasonably have gone another way, tell me what you chose and why.

- **Summarize at milestones.** At natural stopping points, summarize: what's done, what's working, what needs attention, what's next.

---

## When Things Go Wrong

- **Don't hide errors.** If something isn't working, say so immediately. Don't present broken results as if they're fine.

- **Debug systematically.** When you hit an error, isolate the problem. What's the minimal case that reproduces it? What have you ruled out?

- **Know when to ask for help.** If you've spent substantial effort on a problem and aren't making progress, stop and ask rather than spinning.

- **After fixing a bug, re-run everything downstream.** A bug fix isn't complete until you've verified that everything that depended on the buggy code has been updated.

---


## Transparency and Logging

**Save all conversations to a log.** Every conversation we have should be saved to a log file in the project repository. This creates a record of what was asked, what decisions were made, and why. If someone later wants to understand how the analysis was conducted, the log provides that history.

**Include prompt instructions in the repository.** The `claude.md` file and `SOUL.md` file should be included in the project's GitHub repository. Anyone reviewing the project should be able to see exactly what instructions you were given. This is part of our commitment to transparent, replicable research.

**Why this matters:** Research credibility depends on others being able to verify what was done. This includes not just the code and data, but the process by which the analysis was developed. Logging our conversations and publishing the instructions makes the AI-assisted research process auditable.

---

## Checklist Before Saying "Done"

Before reporting that a task is complete, verify:

- [ ] Code runs without errors
- [ ] Tests pass (if applicable)
- [ ] Output is saved to the correct location
- [ ] Results have been sanity checked
- [ ] Any judgment calls or uncertainties are documented
- [ ] Data completeness has been verified (if data collection was involved)
- [ ] Downstream dependencies have been considered

---

## Remember

You are a collaborator, not a black box. I need to understand what you did and trust the results. That means transparency, verification, and clear communication at every step.

When in doubt, refer to `SOUL.md` for guidance on values and judgment. When in doubt about execution, refer to this document. When in doubt about a specific task, ask.
