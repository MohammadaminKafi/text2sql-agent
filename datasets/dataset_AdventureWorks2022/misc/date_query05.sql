-- SH year-only: 1392 = 2013-03-21 .. 2014-03-20 (Gregorian)
USE AdventureWorks2022;
SELECT
  DATETRUNC(month, OrderDate) AS MonthStart,
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE OrderDate >= '2013-03-21' AND OrderDate <= '2014-03-20'
GROUP BY DATETRUNC(month, OrderDate)
ORDER BY MonthStart;