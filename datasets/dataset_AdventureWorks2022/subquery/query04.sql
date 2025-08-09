SELECT v.BusinessEntityID,
       v.Name,
       AVG(poh.TotalDue) AS AvgSpend
FROM Purchasing.Vendor AS v
JOIN Purchasing.PurchaseOrderHeader AS poh ON poh.VendorID = v.BusinessEntityID
GROUP BY v.BusinessEntityID, v.Name
HAVING AVG(poh.TotalDue) >
      (SELECT AVG(TotalDue) FROM Purchasing.PurchaseOrderHeader);