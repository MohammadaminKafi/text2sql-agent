SELECT soh.SalesOrderID,
       soh.OrderDate,
       p.Name AS ProductName,
       sod.OrderQty
FROM Sales.SalesOrderHeader AS soh
JOIN Sales.SalesOrderDetail AS sod
  ON soh.SalesOrderID = sod.SalesOrderID
JOIN Production.Product AS p
  ON sod.ProductID = p.ProductID;