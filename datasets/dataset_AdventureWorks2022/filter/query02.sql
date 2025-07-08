SELECT e.BusinessEntityID, 
       per.FirstName, 
       per.LastName 
FROM HumanResources.Employee AS e 
JOIN Person.Person AS per ON per.BusinessEntityID = e.BusinessEntityID 
WHERE e.BusinessEntityID NOT IN 
      (SELECT BusinessEntityID 
       FROM HumanResources.EmployeePayHistory 
       WHERE Rate > 100);