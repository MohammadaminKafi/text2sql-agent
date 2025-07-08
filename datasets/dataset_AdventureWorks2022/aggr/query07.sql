SELECT c.CustomerID,
       COUNT(soh.SalesOrderID) AS OrderCount,
       SUM(soh.TotalDue) AS TotalSpent
FROM Sales.Customer AS c
JOIN Sales.SalesOrderHeader AS soh
  ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID
HAVING SUM(soh.TotalDue) > (
    SELECT AVG(TotalDue)
    FROM Sales.SalesOrderHeader
)
ORDER BY TotalSpent DESC;