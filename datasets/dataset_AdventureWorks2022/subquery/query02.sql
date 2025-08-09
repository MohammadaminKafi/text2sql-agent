SELECT c.CustomerID,
       COUNT(soh.SalesOrderID) AS OrdersPlaced
FROM Sales.Customer AS c
JOIN Sales.SalesOrderHeader AS soh ON soh.CustomerID = c.CustomerID
GROUP BY c.CustomerID
HAVING COUNT(soh.SalesOrderID) >
      (SELECT AVG(OrderCount)
       FROM (SELECT COUNT(*) AS OrderCount
             FROM Sales.SalesOrderHeader
             GROUP BY CustomerID) AS t);