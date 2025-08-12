Great questions — these two ambiguities (language → many senses, and sense → many tables) are exactly where NL→SQL systems wobble. Here are practical ways to harden your pipeline without making users technical.

# 1) Reduce translation ambiguity (word ↔ senses)

**A. Translate to *n-best* with senses + confidence, not a single string**

* Change `TranslatePromptSig` to return `candidates: List[{text, terms:[{surface, sense_en, confidence}]}]`.
* Ask the LLM: “Produce 3 concise English paraphrases. For each key noun, list possible senses with short glosses and confidences.”
* Downstream stages consume the *set* of senses, not just one translation.

**B. Constrained, schema-aware translation (glossary biasing)**

* Build a glossary from your DB: table/column names, synonyms, tags (see 2C), and common business terms (FA → EN).
* Pass that glossary into the translation prompt: “Prefer these English terms when appropriate; if not applicable, do not force.”
* This keeps “سفارش” aligned to `Order`, “فروش” to `Sales`, etc., when context agrees.

**C. Keep ambiguity explicit**

* Preserve key ambiguous tokens as `{term: "سفارش", senses:["order (sales)","order (purchase)"]}` in the payload you pass into Keyword/Schema/Table matchers. Don’t flatten too early.

**D. Morphology & multiword normalization**

* Persian has compounds and affixes. Normalize (strip ZWNJ/half-spaces, lemmatize common variants) before translation so “سفارشات/سفارش‌ها” converge.

**E. Back-translation sanity check**

* Quick guard: translate FA→EN, then EN→FA. If key nouns drift, flag higher ambiguity and trigger clarification.

# 2) Disambiguate senses → tables/columns

**A. Multilingual semantic grounding over *schema metadata***

* Embed (once) every schema object using multilingual embeddings (names, comments, extended properties, sample values).
* At runtime, embed each *sense* from step 1 and retrieve top-k candidate objects.
* Score = α·name\_similarity + β·comment\_similarity + γ·value\_example\_similarity.

**B. Use data profiling to separate close calls**

* Pull tiny samples & stats per candidate column: example values, distinctness, numeric vs text, date density.
* Match sense expectations: if the sense implies a date, columns with datetime + seasonal distribution get a boost; if monetary, numeric + currency-like value patterns get a boost.

**C. Curate lightweight synonyms on the schema**

* For SQL Server you can mine `sys.extended_properties` (a great place to store human descriptions and synonyms).
* Add per-table/column tags like: `aliases=["order","sales order","purchase order"]`, `fa=["سفارش","سفارش خرید","سفارش فروش"]`.
* This massively improves retrieval without touching data.

**D. Graph-aware scoring (respect join paths)**

* If a sense (e.g., “customer”) maps to multiple tables, prefer those that:

  1. have keys/PKs with that entity in name or tag,
  2. sit on shorter FK paths to other senses in the same query (e.g., customer ↔ order ↔ line).
* Implement: build an FK graph and add a cohesion term: objects that minimize the Steiner tree connecting all senses get higher scores.

**E. Candidate set + thresholding**

* Keep top-k candidates per sense (k≈3). If the score gap between #1 and #2 < δ, treat as *ambiguous* and ask the user a pointed question (see 3). Otherwise, auto-select.

# 3) Ask fewer, better questions (in the user’s language)

You already have `ask_user`. Upgrade the questions:

* Show **2–3 labeled choices** with short Persian descriptions + 1 row example:

  * “با «سفارش» منظور شما کدام است؟

    1. سفارش فروش (SalesOrder) — نمونه: SO12345، تاریخ 2024-05-12
    2. سفارش خرید (PurchaseOrder) — نمونه: PO77821، تاریخ 2024-05-10”
* If multiple columns compete inside the same table, show **data-type and example values**.
* Only ask if *(a)* high entropy across candidates **and** (b) the choice materially changes the result. Otherwise, pick confidently and note the assumption in the summary.

# 4) Make your modules sense-aware (minimal code changes)

Here’s how to weave it into your current pipeline:

**TranslatePrompt → produce n-best + senses**

* Extend `TranslatePromptSig` to output `paraphrases` and `term_senses`.
* Pass `term_senses` to `KeywordExtractor` (or replace it: you already have the nouns).

**Keyword/Schema/Table matchers → accept senses & scores**

* Replace `keyword: str` with `sense: {text, confidence}`.
* Update `MatchSchemas/MatchTables` to run vector search over schema metadata and return `(candidate, score)` lists.

**ColumnSelector → profile-driven boost**

* When multiple columns tie, fetch small stats/samples and re-score using expectations inferred from the sense (date vs id vs money).

**AmbiguityResolver → multiple-choice UX**

* Feed it the scored candidates + 1–row previews to form the options.
* Record the chosen mapping (see 5).

**GenerateSQL → constrained to chosen objects**

* Only allow tables/columns the resolver selected (you already pass context; just prune to confirmed objects).

# 5) Learn from feedback (so you ask less over time)

* **Cache mappings**: `(user_locale="fa", term="سفارش", domain="sales") → {table:"Sales.SalesOrderHeader"}` with confidence.
* Log when users override your default; boost that mapping for similar future prompts (same tenant/domain).
* Periodically export these to your synonym tags (2C).

# 6) Optional power-ups

* **Constrained decoding with a whitelist**: Give the LLM a dynamic list of allowed table/column tokens and ask it to *only* select from those; no hallucinated names.
* **Prompt paraphrasing**: Ask the LLM to produce 2–3 English paraphrases emphasizing different plausible senses; run matching for each and merge scores.
* **Low-resource languages**: keep named entities un-translated (or add a transliteration pass) so product codes/IDs survive translation.

# 7) A thin scoring recipe you can implement fast

For each sense *s* and schema object *o*:

```
score(s,o) =
  0.45 · sim_embed(s.text, o.name+o.comment)
+ 0.20 · name_fuzzy_match(s.text, o.name)
+ 0.15 · type_compatibility(s.expected_type, o.sql_type)
+ 0.10 · value_profile_match(s.expected_pattern, o.sample_values)
+ 0.10 · graph_cohesion_bonus(o, selected_objects_so_far)
```

Pick top-k, threshold by margin, and ask only when needed.

---

If you’d like, I can sketch:

* an updated `TranslatePromptSig` and example prompt to get senses+confidences in one shot,
* a small vector indexer for schema names/comments/sample values,
* and the ambiguity question formatter that shows one-row previews in Persian.
