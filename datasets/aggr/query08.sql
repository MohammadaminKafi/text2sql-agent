SELECT v.BusinessEntityID AS VendorID,
       COUNT(pv.ProductID) AS NumProducts,
       AVG(pv.AverageLeadTime) AS AvgLeadTime
FROM Purchasing.Vendor AS v
JOIN Purchasing.ProductVendor AS pv
  ON v.BusinessEntityID = pv.BusinessEntityID
GROUP BY v.BusinessEntityID
HAVING COUNT(pv.ProductID) = (
    SELECT MAX(ProductCount)
    FROM (
        SELECT BusinessEntityID, COUNT(ProductID) AS ProductCount
        FROM Purchasing.ProductVendor
        GROUP BY BusinessEntityID
    ) AS sub
)
ORDER BY AvgLeadTime;