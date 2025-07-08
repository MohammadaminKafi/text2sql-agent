SELECT p.Name, soh.OrderDate, soh.TotalDue
FROM Sales.SalesOrderHeader AS soh
JOIN Production.Product AS p ON soh.SalesOrderID IN (
    SELECT sod.SalesOrderID
    FROM Sales.SalesOrderDetail AS sod
    WHERE sod.ProductID = p.ProductID
)
WHERE soh.TotalDue > 10000
ORDER BY soh.OrderDate DESC;