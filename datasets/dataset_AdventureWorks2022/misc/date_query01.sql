-- Day-only filter: 1st of any month (Gregorian)
USE AdventureWorks2022;
SELECT
  YEAR(HireDate) AS [Year],
  COUNT(*) AS Hires
FROM HumanResources.Employee
WHERE DAY(HireDate) = 1
GROUP BY YEAR(HireDate)
ORDER BY [Year];