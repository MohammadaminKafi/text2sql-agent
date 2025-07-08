SELECT
  YEAR(SOH.OrderDate) AS OrderYear,
  COUNT(*) AS TotalOrders,
  SUM(SOH.TotalDue) AS TotalRevenue,
  AVG(SOH.Freight) AS AvgFreightPerOrder
FROM Sales.SalesOrderHeader AS SOH
WHERE SOH.Status = 5 -- only shipped orders
GROUP BY YEAR(SOH.OrderDate)
ORDER BY OrderYear DESC;
