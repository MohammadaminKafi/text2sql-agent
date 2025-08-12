-- Month-only filter: January birthdays (Gregorian)
USE AdventureWorks2022;
SELECT
  (YEAR(BirthDate) / 10) * 10 AS BirthDecade,
  COUNT(*) AS Employees
FROM HumanResources.Employee
WHERE MONTH(BirthDate) = 1
GROUP BY (YEAR(BirthDate) / 10) * 10
ORDER BY BirthDecade;