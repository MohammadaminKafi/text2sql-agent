SELECT pi.ProductID,
       p.Name,
       pi.Quantity,
       (SELECT SUM(sod.OrderQty) 
        FROM Sales.SalesOrderDetail AS sod 
        WHERE sod.ProductID = pi.ProductID) AS TotalSold,
       (SELECT MAX(sod.OrderQty) 
        FROM Sales.SalesOrderDetail AS sod 
        WHERE sod.ProductID = pi.ProductID) AS MaxSold
FROM Production.ProductInventory AS pi
JOIN Production.Product AS p ON p.ProductID = pi.ProductID
WHERE pi.Quantity < 
      (SELECT AVG(Quantity) 
       FROM Production.ProductInventory 
       WHERE ProductID = pi.ProductID);