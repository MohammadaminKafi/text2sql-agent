SELECT pi.ProductID,
       p.Name,
       pi.Quantity
FROM Production.ProductInventory AS pi
JOIN Production.Product          AS p  ON p.ProductID = pi.ProductID
WHERE pi.Quantity <
      (SELECT AVG(Quantity)
       FROM Production.ProductInventory
       WHERE ProductID = pi.ProductID);