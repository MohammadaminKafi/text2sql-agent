You are an expert in SQL code writing and dataset generation. The dataset you're tasked to generate data for is a Text-to-SQL assessment dataset. Your job is to generate 5 sample test cases using the following parameters:

- Database Engine: SQL Server (T-SQL)
- Database Name: AdventureWorks2022
- Query Class: single_join
- Difficulty: easy

Output the following 4 items in order (for each test case separately):

1. **Base Query**: A correct SQL query that matches the class and uses relevant tables from the database.
2. **Well-Explained Prompt**: A natural language instruction that clearly describes the query's intent.
3. **Poorly-Explained Prompt**: A vague, informal version of the same request.
4. **Underspecified Prompt**: A version missing key details like filtering, joins, or sorting.

Output format:

**Base Query**  
```sql
[SQL here]
```

**Well-Explained Prompt**
"\[natural language]"

**Poorly-Explained Prompt**
"\[natural language]"

**Underspecified Prompt**
"\[natural language]"

Use table/column names appropriate to the database. Vary tables used across queries when generating in bulk.