SELECT
  p.ProductID,
  p.Name,
  CASE
    WHEN CHARINDEX('bike', LOWER(p.Name)) > 0 THEN 'ContainsBike'
    ELSE 'Other'
  END AS CategoryFlag
FROM Production.Product AS p
WHERE LOWER(p.Name) LIKE '%bike%'
ORDER BY p.Name;