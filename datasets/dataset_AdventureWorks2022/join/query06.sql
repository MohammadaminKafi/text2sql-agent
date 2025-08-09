SELECT pc.Name AS Category,
       COUNT(p.ProductID) AS ProductCount
FROM Production.Product AS p
JOIN Production.ProductSubcategory AS ps
  ON p.ProductSubcategoryID = ps.ProductSubcategoryID
JOIN Production.ProductCategory AS pc
  ON ps.ProductCategoryID = pc.ProductCategoryID
GROUP BY pc.Name;
