SELECT h.CustomerID, COUNT(h.SalesOrderID) AS OrderCount, SUM(h.TotalDue) AS TotalSpent 
FROM Sales.SalesOrderHeader AS h 
GROUP BY h.CustomerID 
HAVING SUM(h.TotalDue) > 197452 
ORDER BY TotalSpent DESC;