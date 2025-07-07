SELECT 
    pr.Name AS ProductName,
    pc.Name AS Category,
    ps.Name AS Subcategory,
    ISNULL(SUM(sod.OrderQty),0) AS TotalQuantitySold
FROM Production.Product AS pr
JOIN Production.ProductSubcategory AS ps
  ON pr.ProductSubcategoryID = ps.ProductSubcategoryID
JOIN Production.ProductCategory AS pc
  ON ps.ProductCategoryID = pc.ProductCategoryID
LEFT JOIN Sales.SalesOrderDetail AS sod
  ON pr.ProductID = sod.ProductID
GROUP BY pr.Name, pc.Name, ps.Name
ORDER BY TotalQuantitySold DESC;