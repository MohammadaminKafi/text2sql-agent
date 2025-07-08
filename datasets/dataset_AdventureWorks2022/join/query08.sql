SELECT e.BusinessEntityID, p.FirstName, p.LastName, d.Name AS Department, sh.Name AS Shift, e.HireDate
FROM HumanResources.Employee AS e
JOIN Person.Person AS p ON e.BusinessEntityID = p.BusinessEntityID
JOIN HumanResources.EmployeeDepartmentHistory AS edh ON e.BusinessEntityID = edh.BusinessEntityID
JOIN HumanResources.Department AS d ON edh.DepartmentID = d.DepartmentID
JOIN HumanResources.Shift AS sh ON edh.ShiftID = sh.ShiftID;