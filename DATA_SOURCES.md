# Data sources and expected local layout

The workflow uses public U.S. patent data from PatentsView. The official platform is https://patentsview.org/.

The local PatentsView directory configured through `paths.patentsview_dir` is expected to contain the tables referenced by the scripts, including patent year and date records, CPC assignments, claim archives, citations, citation counts, patent text, inventor links, assignee links, diversity measures, and text-similarity measures.

The exact release, download date, checksums, and any transformations applied before analysis should be recorded by the user for each rerun. This repository intentionally does not redistribute source data or derived data.

The LLM-extraction stage creates structured P/M/E records from patent title, abstract, and first-claim text. Raw responses and parsed outputs are treated as generated research data and are excluded from version control.
