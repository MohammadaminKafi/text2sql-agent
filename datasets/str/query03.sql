SELECT
  p.ProductID,
  p.Name,
  CASE
    WHEN LOWER(p.Name) LIKE '%mountain%' THEN 'Mountain'
    WHEN LOWER(p.Name) LIKE '%bike%' THEN 'Bike'
    ELSE 'Other'
  END AS ProductType,
  CHARINDEX(' ', p.Name + ' ') AS FirstSpacePos
FROM Production.Product AS p
WHERE p.Name LIKE '% %'
ORDER BY ProductType, p.Name;