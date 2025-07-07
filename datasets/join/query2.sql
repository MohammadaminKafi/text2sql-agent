SELECT v.Name AS VendorName, p.Name AS ProductName, v.CreditRating
FROM Purchasing.ProductVendor AS pv
JOIN Purchasing.Vendor AS v ON pv.BusinessEntityID = v.BusinessEntityID
JOIN Production.Product AS p ON pv.ProductID = p.ProductID
WHERE v.CreditRating <= 3
ORDER BY v.CreditRating ASC;