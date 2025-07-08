SELECT 
    v.BusinessEntityID AS VendorID,
    COUNT(pv.ProductID) AS ProductCount,
    SUM(pv.AverageLeadTime) AS SumLeadTime,
    AVG(pv.AverageLeadTime) AS AvgLeadTime
FROM Purchasing.Vendor AS v
JOIN Purchasing.ProductVendor AS pv
  ON v.BusinessEntityID = pv.BusinessEntityID
GROUP BY v.BusinessEntityID
HAVING COUNT(pv.ProductID) >= 3
ORDER BY AvgLeadTime;
