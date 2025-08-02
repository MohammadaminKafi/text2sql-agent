SELECT YEAR(OrderDate) AS OrderYear, SUM(TotalDue) AS TotalOrders, SUM(TaxAmt) AS TotalTax 
FROM Sales.SalesOrderHeader 
GROUP BY YEAR(OrderDate) 
ORDER BY OrderYear;