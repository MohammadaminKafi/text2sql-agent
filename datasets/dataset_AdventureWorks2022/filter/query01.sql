SELECT c.CustomerID, 
       c.StoreID
FROM Sales.Customer AS c 
WHERE EXISTS 
      (SELECT 1 
       FROM Sales.SalesOrderHeader AS soh 
       WHERE soh.CustomerID = c.CustomerID AND soh.OrderDate > '2011-01-01');