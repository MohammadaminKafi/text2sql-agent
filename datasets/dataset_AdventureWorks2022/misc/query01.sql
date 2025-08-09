SELECT
    soh.CustomerID AS CustomerID
    SUM(soh.TotalDue) AS TotalSpent,
    COUNT(soh.SalesOrderID) AS OrderCount
FROM
    Sales.SalesOrderHeader AS soh
JOIN
    Sales.Customer AS c ON soh.CustomerID = c.CustomerID
GROUP BY
    c.CustomerName
HAVING
    SUM(soh.TotalDue) > 197452
ORDER BY
    TotalSpent DESC;