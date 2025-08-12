-- Month+Day: Dec 31 across all years (Gregorian)
SELECT
  YEAR(ShipDate) AS [Year],
  COUNT(*) AS Shipments
FROM Sales.SalesOrderHeader
WHERE MONTH(ShipDate) = 12 AND DAY(ShipDate) = 31
GROUP BY YEAR(ShipDate)
ORDER BY [Year];