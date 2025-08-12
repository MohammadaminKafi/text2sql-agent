-- SH month-only: Mordad ≈ Jul 23–Aug 22 each year (Gregorian window)
-- Filter rule per year: (month=7 and day>=23) OR (month=8 and day<=22)
USE AdventureWorks2022;
SELECT
  YEAR(OrderDate) AS [Year],
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE (MONTH(OrderDate) = 7 AND DAY(OrderDate) >= 23)
   OR (MONTH(OrderDate) = 8 AND DAY(OrderDate) <= 22)
GROUP BY YEAR(OrderDate)
ORDER BY [Year];