-- Full SH date: 1389-10-09 = 2010-12-30 (Gregorian)
SELECT
  COUNT(*) AS Hires
FROM HumanResources.Employee
WHERE CAST(HireDate AS date) = '2010-12-30';