SELECT p.ProductID,
       p.Name,
       p.ListPrice,
       (SELECT COUNT(*) 
        FROM Sales.SalesOrderDetail AS sod 
        WHERE sod.ProductID = p.ProductID) AS SalesCount,
       (SELECT SUM(sod.LineTotal) 
        FROM Sales.SalesOrderDetail AS sod 
        WHERE sod.ProductID = p.ProductID) AS TotalSales
FROM Production.Product AS p
WHERE p.ListPrice > 
      (SELECT AVG(ListPrice) 
       FROM Production.Product 
       WHERE ProductSubcategoryID = p.ProductSubcategoryID)
ORDER BY TotalSales DESC;