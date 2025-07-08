SELECT v.BusinessEntityID, 
       v.Name 
FROM Purchasing.Vendor AS v 
WHERE NOT EXISTS 
      (SELECT 1 
       FROM Purchasing.PurchaseOrderHeader AS poh 
       WHERE poh.VendorID = v.BusinessEntityID AND poh.TotalDue > 1000);