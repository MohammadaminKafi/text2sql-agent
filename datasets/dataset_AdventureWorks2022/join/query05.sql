SELECT p.Name, i.Quantity, i.LocationID, l.Name AS LocationName
FROM Production.ProductInventory AS i
JOIN Production.Product AS p ON i.ProductID = p.ProductID
JOIN Production.Location AS l ON i.LocationID = l.LocationID
WHERE i.Quantity BETWEEN 100 AND 300
ORDER BY i.Quantity DESC;