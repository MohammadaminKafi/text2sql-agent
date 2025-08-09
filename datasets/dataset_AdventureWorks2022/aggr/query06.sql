SELECT
  c.CustomerID,
  COUNT(soh.SalesOrderID) AS OrderCount,
  SUM(soh.SubTotal) AS TotalSubTotal,
  MAX(soh.TotalDue) AS MaxOrderValue
FROM Sales.Customer AS c
JOIN Sales.SalesOrderHeader AS soh
  ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID
HAVING COUNT(soh.SalesOrderID) > 10
ORDER BY TotalSubTotal DESC;
