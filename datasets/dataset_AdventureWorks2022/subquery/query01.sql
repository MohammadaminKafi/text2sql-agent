SELECT e.BusinessEntityID,
       per.FirstName,
       per.LastName,
       eph.Rate
FROM HumanResources.Employee            AS e
JOIN Person.Person                      AS per ON per.BusinessEntityID = e.BusinessEntityID
JOIN HumanResources.EmployeePayHistory  AS eph ON eph.BusinessEntityID = e.BusinessEntityID
WHERE eph.Rate >
      (SELECT AVG(Rate) FROM HumanResources.EmployeePayHistory);