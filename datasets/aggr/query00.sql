SELECT 
    YEAR(OrderDate) AS OrderYear,
    COUNT(*) AS TotalOrders,
    SUM(SubTotal) AS TotalSubTotal,
    AVG(TaxAmt) AS AvgTax
FROM Sales.SalesOrderHeader
GROUP BY YEAR(OrderDate)
ORDER BY OrderYear;