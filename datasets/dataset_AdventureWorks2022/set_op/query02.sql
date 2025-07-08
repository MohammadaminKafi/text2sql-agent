SELECT sp.BusinessEntityID, per.FirstName, per.LastName
FROM Sales.SalesPerson AS sp
JOIN Person.Person AS per ON per.BusinessEntityID = sp.BusinessEntityID
WHERE sp.BusinessEntityID IN (SELECT SalesPersonID FROM Sales.SalesOrderHeader WHERE TotalDue > 1000)
EXCEPT
SELECT sp.BusinessEntityID, per.FirstName, per.LastName
FROM Sales.SalesPerson AS sp
JOIN Person.Person AS per ON per.BusinessEntityID = sp.BusinessEntityID
WHERE sp.BusinessEntityID IN (SELECT SalesPersonID FROM Sales.SalesOrderHeader WHERE OrderDate < '2012-01-01');