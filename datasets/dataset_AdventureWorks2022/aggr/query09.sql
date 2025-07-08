SELECT c.CustomerID,
       COUNT(soh.SalesOrderID) AS Orders,
       SUM(soh.TotalDue) AS TotalSpent,
       AVG(soh.TotalDue) AS AvgOrderValue
FROM Sales.Customer AS c
JOIN Sales.SalesOrderHeader AS soh
  ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID
HAVING AVG(soh.TotalDue) > (
    SELECT AVG(TotalDue)
    FROM Sales.SalesOrderHeader
    WHERE YEAR(OrderDate) = 2011
)
ORDER BY TotalSpent DESC;