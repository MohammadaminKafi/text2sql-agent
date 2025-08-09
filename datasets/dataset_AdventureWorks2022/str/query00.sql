SELECT
  p.ProductID,
  p.Name AS ProductName,
  LEFT(p.Name, CHARINDEX(' ', p.Name + ' ') - 1) AS FirstWord,
  RIGHT(p.Name, LEN(p.Name) - CHARINDEX(' ', p.Name + ' ')) AS RestOfName
FROM Production.Product AS p
WHERE p.Name LIKE '% %'
ORDER BY p.ProductID;