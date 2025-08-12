-- SH year-only: 1391* = 2012-03-20 .. 2013-03-20 (Gregorian)
USE AdventureWorks2022;
SELECT
  DATETRUNC(month, HireDate) AS MonthStart,
  COUNT(*) AS Hires
FROM HumanResources.Employee
WHERE HireDate >= '2012-03-20' AND HireDate <= '2013-03-20'
GROUP BY DATETRUNC(month, HireDate)
ORDER BY MonthStart;