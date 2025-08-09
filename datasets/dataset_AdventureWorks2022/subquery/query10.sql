SELECT e.BusinessEntityID,
       per.FirstName,
       per.LastName,
       (SELECT AVG(eph.Rate) 
        FROM HumanResources.EmployeePayHistory AS eph 
        WHERE eph.BusinessEntityID = e.BusinessEntityID) AS AvgRate,
       (SELECT MAX(eph.Rate) 
        FROM HumanResources.EmployeePayHistory AS eph 
        WHERE eph.BusinessEntityID = e.BusinessEntityID) AS MaxRate
FROM HumanResources.Employee AS e
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID
WHERE e.BusinessEntityID IN 
      (SELECT DISTINCT BusinessEntityID 
       FROM HumanResources.EmployeePayHistory 
       WHERE Rate > 50);