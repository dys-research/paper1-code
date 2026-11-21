# Validation coverage

The source checks parse all public Python files, check workflow entries and
scan source/configuration/documentation for personal absolute paths and common
literal-credential patterns. The release builder excludes private configuration,
raw data, working outputs, weights, caches and bytecode. Each packaged source
file is recorded in the release ZIP's relative-path SHA-256 manifest.

The unittest suite uses small synthetic fixtures and offline computation. It
covers normalized data and claims preparation, source-span validation, extraction
export to measurement handoff, fixed historical windows, leap-day anniversary
boundaries, missing-history scores, semantic-domain partitions, fitting-only
transformations, vector-ID alignment, six forecasting feature sets, clustered
estimators, tied-score average precision, paired resampling and lexical chronology.

Separate synthetic integration runs exercised measurement assembly through the
final model-ready panel and the exact signed-int64 CPC masks. Those runs used
explicit test vectors. No paid API requests, neural model downloads or full
study-data computation were performed while preparing this distribution.

The tested Python data/analysis versions are recorded in
`constraints-tested.txt`. Neural encoding and CUDA execution require their own
environment validation. The supplied archive did not include the original raw
data, human reference labels or historical model outputs, so this software
validation does not establish reproduction of the manuscript's numerical results.

## Additional manuscript diagnostics validation

The added diagnostic suite checks exact forward/backward anniversary bounds,
same-day exclusions, treatment of negative cosines, window means versus sums,
missing windows/zero denominators, checkpoint reuse, complete versus incomplete
four-endpoint Holm families, separate learners, identifier-based sample joins
and the original common-sample restriction. All 29 tests, including the original
21 tests, passed on Windows. The vector-alignment fixture now closes its
read-only memory map before removing its temporary files.

Using the author's locally retained study inputs, targeted numerical checks
also established:

- The aggregate-input Kelly route reproduces every saved per-patent log mean
  ratio, with 134,532 valid ratios and median 0.6960775766783982.
- Reprocessing the original four-task tree predictions with 500 paired cluster
  resamples reproduces all saved point differences, interval bounds, raw
  sensitivity p-values and Holm-adjusted p-values. The top-5% adjusted p-value
  is 0.0239520958083832.
- The sample generator selects 128,026 patents, compared with 128,027 using
  the existing available-observation rule. Re-estimation of the conditional
  M5 citation and semantic-reuse models reproduces all saved coefficients;
  the conditional-score coefficients are 0.0467393600775213 and
  -0.0275805394001131 respectively.

These are targeted checks against archived study results. The full large-array
Kelly search was checked with synthetic data, including checkpoint reuse; it
was not rerun over the complete corpus. CUDA execution, extraction APIs and
all other manuscript experiments were not rerun. The original tested-version
constraints above are retained; the additional local checks used Python 3.13,
NumPy 2.5.2, pandas 3.0.5, PyArrow 25.0.1, SciPy 1.18.1, scikit-learn 1.9.1
and statsmodels 0.15.0, with no repository dependency changes.

## Internal-link sensitivity branches

Three additional synthetic tests cover secondary/shared assignees, unknown
ownership, historical family membership without future-merge leakage, and family
membership evaluated at the subsequent patent's grant date. They also check
insufficient-history missingness and separate family/assignee/union exclusions.

Targeted checks using locally retained study inputs established:

- Reconstructed temporal-family leaf ranks and intervals match all 191,685
  saved historical-pool records.
- For four real patents in different subfields, all 48 strict-ten-year and
  five-year scores across six representations match the archived results
  within a maximum absolute difference of 0.000000257.
- For three real patents with internal links, all five reuse-count fields
  match exactly after examining their full set of 797,297 future candidates.
- Re-fitting the 18 historical-comparison models and six subsequent-text
  exclusion models from the archived metric inputs reproduces all saved
  coefficients exactly in the checked environment.

The large searches were checked on these selected real queries, not rerun for
every focal patent. CUDA execution was not exercised. These additions keep the
main measures, default workflow and estimators unchanged.
