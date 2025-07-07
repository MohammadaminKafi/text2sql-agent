SELECT c.CustomerID,
       p.FirstName,
       p.LastName,
       SUM(soh.TotalDue) AS TotalSpent
FROM Sales.Customer AS c
JOIN Person.Person AS p
  ON c.PersonID = p.BusinessEntityID
JOIN Sales.SalesOrderHeader AS soh
  ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID, p.FirstName, p.LastName;