SELECT sp.BusinessEntityID,
       per.FirstName,
       per.LastName,
       SUM(soh.TotalDue) AS TotalSales
FROM Sales.SalesPerson  AS sp
JOIN Person.Person      AS per ON per.BusinessEntityID = sp.BusinessEntityID
JOIN Sales.SalesOrderHeader AS soh ON soh.SalesPersonID = sp.BusinessEntityID
GROUP BY sp.BusinessEntityID, per.FirstName, per.LastName
HAVING SUM(soh.TotalDue) >
      (SELECT AVG(TotalSales)
       FROM (SELECT SUM(TotalDue) AS TotalSales
             FROM Sales.SalesOrderHeader
             WHERE SalesPersonID IS NOT NULL
             GROUP BY SalesPersonID) AS q);