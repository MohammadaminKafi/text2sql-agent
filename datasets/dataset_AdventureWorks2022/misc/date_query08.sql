-- Month+Year: 2011/07 (Gregorian)
SELECT
  CAST(StartDate AS date) AS [Date],
  COUNT(*) AS WorkOrders
FROM Production.WorkOrder
WHERE StartDate >= '2011-07-01' AND StartDate < '2011-08-01'
GROUP BY CAST(StartDate AS date)
ORDER BY [Date];