SELECT c.FirstName, c.LastName, a.City, a.PostalCode
FROM Sales.Customer AS cu
JOIN Person.Person AS c ON cu.PersonID = c.BusinessEntityID
JOIN Person.Address AS a ON cu.CustomerID IN (
    SELECT ca.CustomerID
    FROM Sales.CustomerAddress AS ca
    WHERE ca.AddressID = a.AddressID
)
WHERE a.City = 'Seattle';