SELECT e.BusinessEntityID,
       per.FirstName,
       per.LastName,
       e.SickLeaveHours,
       DENSE_RANK() OVER (PARTITION BY e.JobTitle ORDER BY e.SickLeaveHours DESC) AS SickLeaveRank
FROM HumanResources.Employee AS e
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID;