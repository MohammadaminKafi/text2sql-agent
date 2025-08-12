-- Day-only filter: 15th of any month (Gregorian)
SELECT
  YEAR(OrderDate) AS OrderDateYear,
  MONTH(OrderDate) AS OrderDateMonth,
  SUM(TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader
WHERE DAY(OrderDate) = 15
GROUP BY YEAR(OrderDate), MONTH(OrderDate)
ORDER BY YEAR(OrderDate), MONTH(OrderDate);