SELECT
    YEAR(OrderDate) AS OrderYear,
    COUNT(SalesOrderID) AS ShippedOrderCount,
    SUM(TotalDue) AS TotalRevenue,
    AVG(Freight) AS AverageFreightPerOrder
FROM
    Sales.SalesOrderHeader
GROUP BY
    YEAR(OrderDate)
ORDER BY
    OrderYear;