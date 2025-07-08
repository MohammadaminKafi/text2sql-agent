SELECT sp.BusinessEntityID,
       per.FirstName,
       per.LastName,
       (SELECT SUM(soh.TotalDue) 
        FROM Sales.SalesOrderHeader AS soh 
        WHERE soh.SalesPersonID = sp.BusinessEntityID) AS TotalSales,
       (SELECT AVG(soh.TotalDue) 
        FROM Sales.SalesOrderHeader AS soh 
        WHERE soh.SalesPersonID = sp.BusinessEntityID) AS AvgSales
FROM Sales.SalesPerson AS sp
JOIN Person.Person AS per ON per.BusinessEntityID = sp.BusinessEntityID
WHERE sp.BusinessEntityID IN 
      (SELECT DISTINCT SalesPersonID 
       FROM Sales.SalesOrderHeader 
       WHERE TotalDue > 500);