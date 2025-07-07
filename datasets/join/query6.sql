SELECT s.Name AS StoreName, p.FirstName, p.LastName, a.City, sp.Name AS StateProvince
FROM Sales.Customer AS c
JOIN Sales.Store AS s ON c.StoreID = s.BusinessEntityID
JOIN Person.Person AS p ON c.PersonID = p.BusinessEntityID
JOIN Sales.CustomerAddress AS ca ON c.CustomerID = ca.CustomerID
JOIN Person.Address AS a ON ca.AddressID = a.AddressID
JOIN Person.StateProvince AS sp ON a.StateProvinceID = sp.StateProvinceID;