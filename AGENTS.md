# Repository instructions

## Prefer native linuxbox execution

- For data or compute located on `linuxbox`, use the configured SSH host alias
  `linuxbox` and run the work natively there by default. Do not default to
  reading the same data through the Windows `L:` SSHFS drive.
- Treat these as equivalent locations:
  - `L:\Documents\reddit_analyses` ↔ `/home/cara/Documents/reddit_analyses`
  - frozen Study 1 mirror:
    `L:\Documents\reddit_analyses\thread-size` ↔
    `/home/cara/Documents/reddit_analyses/thread-size`
- Prefer native SSH execution especially for recursive discovery, checksums,
  compression/decompression, archive streaming, and scans of large CSV,
  Parquet, pickle, or model-output files. SSHFS may be used for lightweight
  path checks, headers, and small-file reads, or when native SSH is unavailable.
- Do not download or copy large raw archives from `linuxbox` to Windows merely
  to run an analysis. Ask the user before making any such copy.
- Keep source datasets and frozen artefacts read only. When a local repository
  task needs remote computation, use a uniquely named temporary directory under
  `/tmp` on `linuxbox`, return only the required aggregate or derived outputs to
  the authorized local output directory, and remove the remote temporary files
  after successful transfer.
- The saved remote Codex project named `thread-size` points to
  `/home/cara/Documents/reddit_analyses/thread-size`. It is the frozen artefact
  mirror, not this writable Git repository. Do not write audit or code changes
  into that mirror.
- If SSH needs approval, request it. Do not fall back to an expensive bulk
  `L:` read solely to avoid requesting approval.

Windows-local execution remains appropriate for edits and tests confined to
this Git repository, for Windows application integration, or when the user
explicitly requests it.
