SELECT 
    v.Name AS VendorName,
    COUNT(DISTINCT pv.ProductID) AS DistinctProducts,
    SUM(p.ListPrice * pv.StandardPrice) AS PriceSumEstimate
FROM Purchasing.Vendor AS v
JOIN Purchasing.ProductVendor AS pv
  ON v.BusinessEntityID = pv.BusinessEntityID
JOIN Production.Product AS p
  ON pv.ProductID = p.ProductID
GROUP BY v.Name
HAVING COUNT(DISTINCT pv.ProductID) > 5;