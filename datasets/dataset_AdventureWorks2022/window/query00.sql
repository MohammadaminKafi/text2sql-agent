SELECT e.BusinessEntityID,
       per.FirstName,
       per.LastName,
       e.HireDate,
       RANK() OVER (PARTITION BY e.JobTitle ORDER BY e.HireDate DESC) AS RankByHireDate
FROM HumanResources.Employee AS e
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID;