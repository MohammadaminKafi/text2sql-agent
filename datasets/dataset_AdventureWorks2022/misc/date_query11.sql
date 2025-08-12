-- Month+Day: Feb 14 across all years (Gregorian)
SELECT
  YEAR(OrderDate) AS [Year],
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE MONTH(OrderDate) = 2 AND DAY(OrderDate) = 14
GROUP BY YEAR(OrderDate)
ORDER BY [Year];