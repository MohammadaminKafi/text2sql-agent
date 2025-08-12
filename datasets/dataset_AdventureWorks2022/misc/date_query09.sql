-- SH Month+Year: Aban 1392 = 2013-10-23 .. 2013-11-21 (Gregorian)
SELECT
  CAST(OrderDate AS date) AS [Date],
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE OrderDate >= '2013-10-23' AND OrderDate <= '2013-11-21'
GROUP BY CAST(OrderDate AS date)
ORDER BY [Date];