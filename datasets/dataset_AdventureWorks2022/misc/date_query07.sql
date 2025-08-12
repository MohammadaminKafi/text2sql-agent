-- Month+Year: March 2013 = 2013-03-01 .. 2013-03-31 (Gregorian)
SELECT
  CAST(OrderDate AS date) AS [Date],
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE OrderDate >= '2013-03-01' AND OrderDate < '2013-04-01'
GROUP BY CAST(OrderDate AS date)
ORDER BY [Date];