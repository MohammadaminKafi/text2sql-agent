SELECT e.BusinessEntityID, per.FirstName, per.LastName
FROM HumanResources.Employee AS e
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID
WHERE e.HireDate < '2020-01-01'
EXCEPT
SELECT e.BusinessEntityID, per.FirstName, per.LastName
FROM HumanResources.Employee AS e
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID
WHERE e.JobTitle = 'Manager';