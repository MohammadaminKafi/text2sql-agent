-- Full Gregorian date by name: April 16, 2011
SELECT
  SUM(TaxAmt) AS TotalTax
FROM Purchasing.PurchaseOrderHeader
WHERE CAST(OrderDate AS date) = DATEFROMPARTS(2011, 4, 16);