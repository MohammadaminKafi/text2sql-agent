-- Month-only filter: July (7) across all years (Gregorian)
USE AdventureWorks2022;
SELECT
  YEAR(OrderDate) AS [Year],
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE MONTH(OrderDate) = 7
GROUP BY YEAR(OrderDate)
ORDER BY [Year];