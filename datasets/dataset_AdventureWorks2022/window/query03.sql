SELECT sp.BusinessEntityID,
       per.FirstName,
       per.LastName,
       SUM(soh.TotalDue) AS TotalSales,
       ROW_NUMBER() OVER (PARTITION BY sp.BusinessEntityID ORDER BY SUM(soh.TotalDue) DESC) AS SalesRank
FROM Sales.SalesPerson AS sp
JOIN Person.Person AS per ON per.BusinessEntityID = sp.BusinessEntityID
JOIN Sales.SalesOrderHeader AS soh ON soh.SalesPersonID = sp.BusinessEntityID
GROUP BY sp.BusinessEntityID, per.FirstName, per.LastName;