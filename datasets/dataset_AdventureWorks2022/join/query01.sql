SELECT p.FirstName, p.LastName, e.JobTitle, e.HireDate
FROM HumanResources.Employee AS e
JOIN Person.Person AS p ON e.BusinessEntityID = p.BusinessEntityID
WHERE e.HireDate < '2015-01-01'
AND e.JobTitle LIKE '%Manager%';