SELECT p.FirstName, p.LastName, a.AddressLine1, a.City, sp.Name AS StateProvince, cr.Name AS CountryRegion
FROM Person.Person AS p
JOIN Person.BusinessEntityAddress AS bea ON p.BusinessEntityID = bea.BusinessEntityID
JOIN Person.Address AS a ON bea.AddressID = a.AddressID
JOIN Person.StateProvince AS sp ON a.StateProvinceID = sp.StateProvinceID
JOIN Person.CountryRegion AS cr ON sp.CountryRegionCode = cr.CountryRegionCode;