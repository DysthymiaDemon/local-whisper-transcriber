# AGENTS.md

## Instructions

- Always start and keep caveman mode on.
- If caveman mode was off at the start of a new chat, report that status.
- When the user asks for `git add commit push`, stage relevant repo changes, commit with an appropriate summary and description, and push the current branch without asking again.
- If normal `git push` cannot prompt for credentials, use the PAT stored in Windows Credential Manager target `GitHub - https://api.github.com/DysthymiaDemon` via an in-memory `http.extraHeader` push. Do not print, commit, or persist the token.
