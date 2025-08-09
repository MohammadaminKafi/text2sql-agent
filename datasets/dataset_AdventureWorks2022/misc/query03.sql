SELECT
    soh.TerritoryID AS Territory,
    COUNT(DISTINCT c.CustomerID) AS NumberOfDistinctCustomers,
    SUM(soh.Freight) AS TotalFreightAmount,
    AVG(soh.Freight) AS AverageFreightAmount
FROM
    Sales.SalesOrderHeader AS soh
JOIN
    Sales.Customer AS c ON soh.CustomerID = c.CustomerID
GROUP BY
    soh.TerritoryID
ORDER BY
    TotalFreightAmount DESC;