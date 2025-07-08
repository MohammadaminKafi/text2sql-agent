Organized from simple to complex to help assess model capabilities across increasing levels of SQL complexity.

### Basic Queries (dir: basic)
- `basic_select`: Single-table SELECT queries without WHERE clause.
- `conditional_select`: SELECT with simple WHERE conditions.

### Joins (dir: join)
- `single_join`: Query with one JOIN between two related tables.
- `double_join`: JOIN between three tables.
- `multi_join`: Multiple JOINs across four or more tables.

### Aggregation (dir: aggr)
- `basic_aggregate`: COUNT, SUM, AVG with GROUP BY.
- `advanced_aggregate`: HAVING clauses, nested aggregates, multi-group levels.

### Sorting and Limiting (dir: sort_limit)
- `sorting`: ORDER BY (ASC/DESC).
- `top_n`: Top N records using LIMIT, TOP, FETCH, or ROW_NUMBER.

### String/Text Handling (dir: str)
- `string_functions`: LIKE, UPPER/LOWER, LENGTH, CHARINDEX, etc.
- `advanced_string`: SUBSTRING, REPLACE, CONCAT, and pattern matching.

### Filtering (dir: filter)
- `in_filtering`: IN/NOT IN with subqueries or value sets.
- `exists_filtering`: EXISTS/NOT EXISTS clauses.
- `null_checking`: IS NULL / IS NOT NULL.

### Subqueries (dir: subquery)
- `simple_subquery`: Subqueries inside SELECT or WHERE.
- `correlated_subquery`: Subquery referencing outer query.
- `scalar_subquery`: Single-value subquery.

### Window Functions (dir: window)
- `basic_window`: RANK, ROW_NUMBER over PARTITION BY.
- `complex_window`: Window frames and complex partitions.

### Set Operations (dir: set_op)
- `union_intersect_except`: Queries using UNION, INTERSECT, EXCEPT.

### Date (dir: date)
