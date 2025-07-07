SELECT soh.SalesOrderID, p.FirstName, p.LastName, a.City, sp.Name AS StateProvince, cr.Name AS Country
FROM Sales.SalesOrderHeader AS soh
JOIN Sales.Customer AS c ON soh.CustomerID = c.CustomerID
JOIN Person.Person AS p ON c.PersonID = p.BusinessEntityID
JOIN Sales.CustomerAddress AS ca ON c.CustomerID = ca.CustomerID
JOIN Person.Address AS a ON ca.AddressID = a.AddressID
JOIN Person.StateProvince AS sp ON a.StateProvinceID = sp.StateProvinceID
JOIN Person.CountryRegion AS cr ON sp.CountryRegionCode = cr.CountryRegionCode;