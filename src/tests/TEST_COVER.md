# Test Coverage Tracker

- Update these tables as you add tests.
- **Status**: ✅ has tests · ❌ missing tests
- Use **TODO** to note gaps, flakiness, data deps, or next steps.

## Files

| Status | Source File                      | Primary Test File(s)                                 | Markers             | TODO                                  |
| ------ | -------------------------------- | ---------------------------------------------------- | ------------------- | ------------------------------------- |
| ❌    | `components/utils/date_utils.py` | `tests/utils/test_date_utils.py`                     |                     |                                       |
| ❌    | `components/utils/helpers.py`    | `tests/utils/test_helpers.py`                        |                     |                                       |
| ❌    | `components/utils/llm_utils.py`  | `tests/llm/test_llm_utils_llm.py`                    |                     |                                       |
| ❌    | `components/utils/vis_utils.py`  | `tests/utils/test_vis_utils.py`                      |                     |                                       |
| ❌    | `components/dspy_signatures.py`  | see **Signatures** table                             |                     |                                       |
| ❌    | `components/dspy_modules.py`     | `tests/llm/test_dspy_modules_llm.py`                 |                     |                                       |
| ❌    | `components/dspy_tools.py`       | `tests/llm/test_dspy_tools_llm.py`                   |                     |                                       |
| ❌    | `components/top_flows.py`        | `tests/test_end2end.py`, `tests/test_integration.py` |                     |                                       |
| ❌    | `components/logging_setup.py`    |                                                      |                     |                                       |
| ❌    | `components/types/vis_types.py`  |                                                      |                     |                                       |
| ❌    | `main.py`                        |                                                      |                     |                                       |

---

## Functions

| Status | Function                      | Defined In                       | Test File                         | Markers | TODO                         |
| ------ | ----------------------------- | -------------------------------- | --------------------------------- | ------- | ---------------------------- |
| ❌     |                               |                                  |                                   |         |                              |


---

## Signatures

| Status | Signature                    | Test File                                             | Markers            | TODO                           |
| ------ | ---------------------------- | ----------------------------------------------------- | ------------------ | ------------------------------ |
| ✅      | `QuickGateSig`               | `tests/llm/test_quick_gate_sig_llm.py`                | `llm`, `signature` |                               |
| ✅      | `ExtractDatesSig`            | `tests/llm/test_extract_dates_sig_llm.py`             | `llm`, `signature` |                               |
| ✅      | `NormalizeDatesTranslateSig` | `tests/llm/test_normalize_dates_translate_sig_llm.py` | `llm`, `signature` |                               |
| ✅      | `SqlReadyPromptSig`          | `tests/llm/test_sql_ready_prompt_sig_llm.py`          | `llm`, `signature` |                               |
| ✅      | `DetectAmbiguitySig`         | `tests/llm/test_detect_ambiguity_sig_llm.py`          | `llm`, `signature` |                               |
| ❌      |                   |                                                       |                    |                               |

---

## Modules

| Status | Module/Class                 | Defined In                   | Test File                            | Markers | TODO                      |
| ------ | ---------------------------- | ---------------------------- | ------------------------------------ | ------- | ------------------------- |
| ❌     |                             |                               |                                      |         |                           |

---

## Flows (integration / end-to-end)

---

## Golden / Regression Sets
