# Golden eval report

- **Result:** ❌ FAIL
- **Cases:** 16/22 passed
- **Infrastructure (rate-limit) failures:** 4 — transient, not a quality regression; re-run these.
- **Quality failures:** 2 — genuine regressions.
- **Tokens (this run):** ~149,148
- **Duration:** 1217.7s

## curriculum

| Case | Result | Key scores |
| --- | --- | --- |
| python_web_dev | ✅ | module_count=13, ordering=0.0667, coverage_judge=0.8 |
| thermodynamics | ✅ | module_count=15, ordering=0.2857, coverage_judge=0.875 |
| sql_for_analysts | ✅ | module_count=15, ordering=0.7857, coverage_judge=0.815 |
| do_not_include_heavy | ✅ | module_count=11, ordering=0.5, coverage_judge=0.8 |
| known_concepts_heavy | ✅ | module_count=15, ordering=0.4333, coverage_judge=0.825 |
| calculus_beginner | ✅ | module_count=15, ordering=0.2308, coverage_judge=0.85 |
| react_frontend | ❌ | module_count=14, ordering=0.5556, coverage_judge=0.9 |
| dsa_interview | ❌ | module_count=29, ordering=0.8, coverage_judge=0.85 |
  - ❌ `react_frontend` — all must_include present (missing=['Hooks'])
  - ❌ `dsa_interview` — all must_include present (missing=['Arrays', 'Trees'])

## diagnosis

| Case | Result | Key scores |
| --- | --- | --- |
| clearly_correct | ✅ | mastery_signal=clear |
| clearly_wrong | ✅ | mastery_signal=weak |
| vague | ⚠️ infra |  |
| confident_but_wrong | ⚠️ infra |  |
| injection_adversarial | ⚠️ infra |  |
| partial_correct | ✅ | mastery_signal=weak |
| correct_code_trace | ✅ | mastery_signal=clear |
| misconception | ✅ | mastery_signal=weak |
| dont_know | ✅ | mastery_signal=weak |
| detailed_correct_math | ⚠️ infra |  |
  - ❌ `vague` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `confident_but_wrong` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `injection_adversarial` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `detailed_correct_math` — ran-without-error (diagnosis fell back (likely rate limit/timeout))

## lesson

| Case | Result | Key scores |
| --- | --- | --- |
| python_functions_medium | ✅ | word_count=674, quality_judge=0.8742 |
| thermo_first_law_deep | ✅ | word_count=1010, quality_judge=0.9401 |
| sql_joins_fast | ✅ | word_count=424, quality_judge=0.9125 |
| french_revolution_medium | ✅ | word_count=835, quality_judge=0.888 |
