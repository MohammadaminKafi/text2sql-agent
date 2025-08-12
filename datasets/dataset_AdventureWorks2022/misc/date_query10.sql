-- SH Month+Year: Ordibehesht 1391 to Ordibehesht 1392 = 2012-04-20 .. 2013-05-20 (Gregorian)
SELECT
  CAST(OrderDate AS date) AS [Date],
  SUM(SubTotal) AS PurchaseSubTotal
FROM Purchasing.PurchaseOrderHeader
WHERE OrderDate >= '2012-04-20' AND OrderDate <= '2013-05-20'
GROUP BY CAST(OrderDate AS date)
ORDER BY [Date];