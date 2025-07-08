SELECT 
    c.CustomerID,
    COUNT(soh.SalesOrderID) AS NumOrders,
    SUM(soh.TotalDue) AS TotalSpent
FROM Sales.Customer AS c
JOIN Sales.SalesOrderHeader AS soh
  ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID
HAVING SUM(soh.TotalDue) > 5000
ORDER BY TotalSpent DESC;