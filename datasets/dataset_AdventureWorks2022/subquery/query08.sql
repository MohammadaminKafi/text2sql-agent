SELECT sod.SalesOrderID,
       sod.ProductID,
       sod.OrderQty
FROM Sales.SalesOrderDetail AS sod
WHERE sod.OrderQty >
      (SELECT AVG(sod2.OrderQty)
       FROM Sales.SalesOrderDetail AS sod2
       WHERE sod2.ProductID = sod.ProductID);