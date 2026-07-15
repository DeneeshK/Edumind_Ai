# Golden eval report

- **Result:** ❌ FAIL
- **Cases:** 10/19 passed
- **Infrastructure (rate-limit) failures:** 9 — transient, not a quality regression; re-run these.
- **Tokens (this run):** ~27,927
- **Duration:** 1608.7s

## curriculum

| Case | Result | Key scores |
| --- | --- | --- |
| python_web_dev | ⚠️ infra |  |
| thermodynamics | ⚠️ infra |  |
| do_not_include_heavy | ⚠️ infra |  |
| known_concepts_heavy | ⚠️ infra |  |
| dsa_interview | ⚠️ infra |  |
  - ❌ `python_web_dev` — ran-without-error (Groq rate limit exceeded after 3 retries)
  - ❌ `thermodynamics` — ran-without-error (Groq rate limit exceeded after 3 retries)
  - ❌ `do_not_include_heavy` — ran-without-error (Groq rate limit exceeded after 3 retries)
  - ❌ `known_concepts_heavy` — ran-without-error (Groq rate limit exceeded after 3 retries)
  - ❌ `dsa_interview` — ran-without-error (Groq rate limit exceeded after 3 retries)

## diagnosis

| Case | Result | Key scores |
| --- | --- | --- |
| clearly_correct | ✅ | mastery_signal=clear |
| clearly_wrong | ✅ | mastery_signal=weak |
| vague | ✅ | mastery_signal=uncertain |
| confident_but_wrong | ✅ | mastery_signal=weak |
| injection_adversarial | ⚠️ infra |  |
| partial_correct | ✅ | mastery_signal=weak |
| correct_code_trace | ⚠️ infra |  |
| misconception | ✅ | mastery_signal=weak |
| dont_know | ⚠️ infra |  |
| detailed_correct_math | ⚠️ infra |  |
  - ❌ `injection_adversarial` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `correct_code_trace` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `dont_know` — ran-without-error (diagnosis fell back (likely rate limit/timeout))
  - ❌ `detailed_correct_math` — ran-without-error (diagnosis fell back (likely rate limit/timeout))

## lesson

| Case | Result | Key scores |
| --- | --- | --- |
| python_functions_medium | ✅ | word_count=645, quality_judge=0.9509 |
| thermo_first_law_deep | ✅ | word_count=924, quality_judge=0.9226 |
| sql_joins_fast | ✅ | word_count=390, quality_judge=0.9103 |
| french_revolution_medium | ✅ | word_count=735, quality_judge=0.923 |
