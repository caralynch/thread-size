# Provenance record

## Bottom line

The recovered evidence supports an earlier, separate r/politics collection and
preliminary analysis. In particular, an executed notebook dated 9 May 2022
contains the account-activity calculation later described conceptually by the
repeat-account criterion. It does **not**, however, support the stronger
chronology that the three published criteria were calculated for r/politics,
r/CryptoCurrency, and r/Conspiracy before the latter study-specific datasets
were assembled.

The first recovered comparative data summary is dated 16 January 2023, after
the October 2022 r/Conspiracy and r/CryptoCurrency data period. It covers
r/books, r/Conspiracy, r/CryptoCurrency, and r/TheDonald. Git records r/politics
being added to the combined datasets on 7 February 2023. The workbook first
exists on 29 May 2023, and its `Author thresholds` sheet is first present in a
Git state dated 1 August 2023. The three criteria first appear together in
located paper prose on 17 September 2024.

No dated pre-collection screening table, query, notebook output, 4CAT job
receipt, screenshot, or cited external-statistics source covering all three
selected subreddits was recovered. Preliminary empirical data were demonstrably
used for r/politics before the later 4CAT setup, but whether that work was used
to select the three final subreddits remains **unresolved**. No positive evidence
of external-statistics use was recovered.

## Evidence timeline

| Date | Recovered evidence | Provenance implication |
|---|---|---|
| 18 Nov 2021 | r/politics ZIP entry timestamp for `politics_nov2020.csv` | r/politics raw data existed well before the other two study exports. |
| 7 Apr 2022 | `politics_cleaned_data.csv` timestamp | A separately prepared r/politics dataset existed before October 2022. |
| 9 May 2022 | Executed `descriptive_stats.ipynb` on L: | Recovered original preliminary r/politics account-activity calculation: 239,895 of 587,126 observable accounts had one contribution, so 59.1408% had at least two. The notebook does not describe subreddit selection. |
| 15 Jun 2022 | Three r/politics author-aggregate CSVs on L: | Their schemas, checksums and aggregate counts were recovered. The overall file reproduces the May notebook and cleaned-data result, not the later workbook percentage. |
| 13–18 Oct 2022 | 4CAT source tree and installation note on L: | The note records successful Docker Compose startup and web-interface login on 18 October. This establishes collection capability, not the date or criteria of subreddit selection. |
| 1–30 Oct 2022 | Actual UTC period represented in r/Conspiracy and r/CryptoCurrency exports | This is the records' observation window, not a recovered retrieval/job date. |
| 7 Dec 2022 | r/CryptoCurrency ZIP-entry timestamp | Strong evidence that the export existed by this date. |
| 7 Dec 2022 | Final timestamp of the archived Docker Desktop disk | Read-only inspection found no surviving 4CAT container, named-volume or job metadata in its final filesystem state. |
| 16 Jan 2023 | First `thread_size_prediction` Git commit, including aggregate summaries and cleaning rules | First recovered combined comparison; it contains four candidate subreddits but not politics. |
| 26 Jan 2023 | “finished prelim thread analyses” Git commit | Preliminary analysis was complete after the study exports existed. |
| 7 Feb 2023 | “added r/politics to datasets” Git commit | First recovered combined-study evidence including politics. |
| 29 May 2023 | First XLSX summary workbook in Git | Contains cleaned summaries and thread data, but no `Author thresholds` sheet. |
| 21 Jun 2023 | Commit titled “added author thresholding” | The committed `20_add_author_thresholds.ipynb` body is actually a regression notebook; it neither calculates thresholds nor writes the workbook. The label is not reproducible evidence. |
| 1 Aug 2023 | First Git workbook state with `Author thresholds` | Earliest recovered threshold artifact. Values are stored constants, not formulas. |
| 25 Sep 2023 | `main_results.md` | Notes that politics summaries may need rerunning and asks “what author thresholds used?”, documenting uncertainty about the method. |
| 17 Sep 2024 | First located paper draft with all three criteria | First recovered prose frames five already gathered subreddit datasets as a selection pool. This is later than collection and modelling. |
| 11 Feb 2025 | First commit of the current paper repository | Carries the later claim forward; it does not establish its origin. |
| 13 Nov 2025 | First commit of `thread-size-1` | This repository is later modelling/publication code and cannot date original selection. |

## Source classification

- **Preliminary analysis:** the May–June 2022 r/politics notebook and
  account-level aggregate CSVs. These establish early empirical work on
  r/politics, including repeat-account activity, but do not identify a
  three-subreddit screening decision. The January 2023 summary CSV/ODS is the
  first recovered multi-subreddit comparison and postdates the export periods.
- **Raw collected data:** the three historical 4CAT/Pushshift ZIP/CSV exports.
- **Cleaned study data:** `politics_cleaned_data.csv`, the cleaned summary
  sheets, and safely identified prepared data. The untrusted pickle family was
  not opened.
- **Later modelling data/code:** regression/thread pickles, modelling
  notebooks, `thread-size-1`, and the later paper repositories.
- **Collection setup records:** the 4CAT installation note/tree and archived
  Docker Desktop disk. The installation note records successful setup, but no
  study-specific 4CAT job log or Pushshift query receipt was recovered.
- **Later prose:** the September 2024 paper draft, current manuscript, and 2026
  thesis briefing.

The detailed path-level inventory, including checksums and access decisions, is
in `source_inventory.csv` and `outputs/source_inventory_computed.csv`.

## What the workbook establishes

The `Author thresholds` sheet reports the percentage of accounts with one
“activity”: politics 43.3149%, crypto 46.9019%, conspiracy 42.3210%, and books
68.0454%. Their complements exceed 50% for the three selected subreddits and
fall below 50% for books.

This is a **recovered original summary artifact**, not a recovered original
calculation. The sheet contains values rather than formulas, does not state the
period or denominator, and has no traceable calculation notebook. Its first
dated appearance is after all three study datasets existed.

The May 2022 notebook, June 2022 overall-author CSV, and cleaned r/politics CSV
all imply the same full-period result. The notebook's saved aggregate output
records 239,895 single-contribution accounts among 587,126 observable account
labels, leaving 347,231 accounts with at least two contributions (59.1408%).
The cleaned r/politics CSV also reproduces the workbook's dataset totals exactly:
6,428,330 contributions, 65,343 roots, and 587,126 unique observable account
labels. This differs from the workbook-implied 56.6851%, so the dated
preliminary calculation still does not recover the workbook's later rule.

## Exact missing evidence

The following would be required to claim reproduction of the original
selection process or its pre-collection chronology:

1. a dated artifact predating the October 2022 r/Conspiracy and
   r/CryptoCurrency collection that evaluates the candidate subreddits;
2. the source population or external-statistics citations used for that
   screening, including its date range and retrieval date;
3. the query, formula, script, or notebook output that operationalised “over
   1,000 monthly posts” and stated whether it meant every month, a typical
   month, or a mean;
4. the original post-format field and classification rule, including treatment
   of external links, galleries, and media hosted outside Reddit;
5. the calculation behind the workbook's “% authors with 1 activity” values,
   including its denominator, window, and whether activities were combined,
   monthly, or rolling;
6. the politics-specific deleted, bot, moderator, and service-account handling;
7. the original 4CAT/Pushshift job records or collection receipts for the five
   later-described candidate datasets. The surviving source tree, installation
   note, Docker logs and archived Docker filesystem did not contain them; and
8. evidence linking the recovered May–June 2022 r/politics activity analysis to
   the later decision to retain r/politics, r/Conspiracy and r/CryptoCurrency.

The local 19.3 MB `subreddit_data.zip` copy is not a substitute: it lacks a ZIP
central directory and cannot be read as a complete archive.

## Conclusion classification

| Conclusion | Classification |
|---|---|
| r/politics data and preliminary analysis existed before the October 2022 collection of the other two selected subreddits. | **Recovered original chronology evidence** (not itself a selection calculation). |
| A repeat-account activity calculation was performed on the preliminary r/politics data by May 2022. | **Recovered original preliminary calculation** (not a recovered subreddit-selection calculation). |
| The three criteria were applied to all three selected subreddits before study-specific collection. | **Unresolved.** |
| Preliminary r/politics data were used in the eventual subreddit-selection decision. | **Unresolved; availability and analysis are recovered, decision use is not.** |
| External statistics were used for subreddit screening. | **Unresolved; no positive source evidence recovered.** |
| The August 2023 workbook records complements above 50% for the three selected subreddits. | **Recovered original summary artifact; original calculation unresolved.** |
| The historical exports satisfy the newly defined full-period repeat-account rule. | **Sensitivity analysis under a newly defined rule.** |
| The historical exports are predominantly not image/video under both audit rules and worst-case treatment of unknown format. | **Sensitivity analysis under a newly defined rule.** |
| The historical exports are high-volume over their observed windows; only r/politics contains an inferred-complete calendar month. | **Retrospective validation using later study data.** |

No conclusion in this audit qualifies as a **recovered original selection
calculation**, because no dated calculation with its rules and source population
was recovered.
