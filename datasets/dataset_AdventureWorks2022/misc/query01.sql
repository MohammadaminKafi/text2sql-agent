SELECT
    soh.CustomerID AS CustomerID,
    SUM(soh.TotalDue) AS TotalSpent,
    COUNT(soh.SalesOrderID) AS OrderCount
FROM
    Sales.SalesOrderHeader AS soh
GROUP BY
    soh.CustomerID
HAVING
    SUM(soh.TotalDue) > 197452
ORDER BY
    TotalSpent DESC;