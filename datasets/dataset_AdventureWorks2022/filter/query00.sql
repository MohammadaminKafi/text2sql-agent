SELECT p.ProductID,
       p.Name
FROM Production.Product AS p
WHERE NOT EXISTS (
    SELECT 1
    FROM Sales.SalesOrderDetail AS sod
    WHERE sod.ProductID = p.ProductID
);