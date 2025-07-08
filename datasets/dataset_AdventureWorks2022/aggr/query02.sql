SELECT 
    pc.Name AS Category,
    COUNT(DISTINCT p.ProductID) AS ProductCount,
    AVG(p.ListPrice) AS AvgPrice,
    MAX(p.ListPrice) AS MaxPrice
FROM Production.Product AS p
JOIN Production.ProductSubcategory AS ps
  ON p.ProductSubcategoryID = ps.ProductSubcategoryID
JOIN Production.ProductCategory AS pc
  ON ps.ProductCategoryID = pc.ProductCategoryID
GROUP BY pc.Name
ORDER BY AvgPrice DESC;
