SELECT pi.ProductID,
       p.Name,
       pi.Quantity,
       (SELECT MAX(pi2.Quantity) 
        FROM Production.ProductInventory AS pi2 
        WHERE pi2.ProductID = pi.ProductID) AS MaxInventory,
       (SELECT MIN(pi2.Quantity) 
        FROM Production.ProductInventory AS pi2 
        WHERE pi2.ProductID = pi.ProductID) AS MinInventory
FROM Production.ProductInventory AS pi
JOIN Production.Product AS p ON p.ProductID = pi.ProductID
WHERE pi.Quantity > 
      (SELECT AVG(Quantity) 
       FROM Production.ProductInventory 
       WHERE ProductID = pi.ProductID);