SELECT
    ps.Name AS CategoryName,
    AVG(pl.ListPrice) AS AverageListPrice,
    MAX(pl.ListPrice) AS MaximumListPrice
FROM
    Production.Product p
JOIN
    Production.ProductSubcategory ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID
JOIN
    Production.ProductListPriceHistory pl ON p.ProductID = pl.ProductID
GROUP BY
    ps.Name
ORDER BY
    AverageListPrice ASC;