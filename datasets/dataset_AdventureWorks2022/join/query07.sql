SELECT edh.BusinessEntityID,
       p.FirstName,
       p.LastName,
       d.Name AS DepartmentName
FROM HumanResources.EmployeeDepartmentHistory AS edh
JOIN HumanResources.Department AS d
  ON edh.DepartmentID = d.DepartmentID
JOIN HumanResources.Employee AS e
  ON edh.BusinessEntityID = e.BusinessEntityID
JOIN Person.Person AS p
  ON e.BusinessEntityID = p.BusinessEntityID
WHERE edh.EndDate IS NULL;