SELECT p.Name,
       p.ListPrice,
       ROW_NUMBER() OVER (PARTITION BY p.ProductSubcategoryID ORDER BY p.ListPrice DESC) AS ProductRank
FROM Production.Product AS p;
