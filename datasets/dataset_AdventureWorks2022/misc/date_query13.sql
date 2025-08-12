-- SH Month+Day (Nowruz): 1 Farvardin
-- Mapped dates: 1390→2011-03-21, 1391→2012-03-20, 1392→2013-03-21, 1393→2014-03-21
SELECT
  YEAR(TransactionDate) AS [Year],
  COUNT(*) AS TxnCount
FROM Production.TransactionHistory
WHERE CAST(TransactionDate AS date) IN ('2011-03-21','2012-03-20','2013-03-21','2014-03-21')
GROUP BY YEAR(TransactionDate)
ORDER BY [Year];