SELECT p.ProductID,
       p.Name,
       SUM(sod.LineTotal) AS TotalSales,
       RANK() OVER (PARTITION BY p.ProductSubcategoryID ORDER BY SUM(sod.LineTotal) DESC) AS SalesRank
FROM Sales.SalesOrderDetail AS sod
JOIN Production.Product AS p ON p.ProductID = sod.ProductID
GROUP BY p.ProductID, p.Name, p.ProductSubcategoryID;
