SELECT sp.BusinessEntityID, 
       per.FirstName, 
       per.LastName 
FROM Sales.SalesPerson AS sp 
JOIN Person.Person AS per ON per.BusinessEntityID = sp.BusinessEntityID 
WHERE EXISTS 
      (SELECT 1 
       FROM Sales.SalesOrderHeader AS soh 
       WHERE soh.SalesPersonID = sp.BusinessEntityID AND soh.TotalDue > 1000);