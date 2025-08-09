SELECT p.ProductID, p.Name
FROM Production.Product AS p
WHERE p.ListPrice > 1000
UNION
SELECT p.ProductID, p.Name
FROM Production.Product AS p
WHERE p.Color = 'Red';
