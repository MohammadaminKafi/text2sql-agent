SELECT
  p.ProductID,
  p.Name,
  STUFF((
    SELECT '; ' + CAST(sop.StandardPrice AS VARCHAR(10))
    FROM Purchasing.ProductVendor AS sop
    WHERE sop.ProductID = p.ProductID
    FOR XML PATH(''), TYPE
  ).value('.', 'NVARCHAR(MAX)'), 1, 2, '') AS PricesList
FROM Production.Product AS p
JOIN Purchasing.ProductVendor AS pv ON p.ProductID = pv.ProductID
GROUP BY p.ProductID, p.Name
HAVING COUNT(DISTINCT pv.BusinessEntityID) > 1;