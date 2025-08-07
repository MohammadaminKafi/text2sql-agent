SELECT pv.BusinessEntityID AS VendorID,
    COUNT(pv.ProductID) AS NumberOfProducts,
    SUM(pv.AverageLeadTime) AS TotalLeadTime,
    AVG(pv.AverageLeadTime) AS AverageLeadTime
FROM 
    Purchasing.ProductVendor pv
GROUP BY
    pv.BusinessEntityID
HAVING 
    COUNT(pv.ProductID) >= 3
ORDER BY 
    NumberOfProducts DESC;