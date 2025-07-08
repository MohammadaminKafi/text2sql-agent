SELECT p.ProductID, 
       p.Name 
FROM Production.Product AS p 
WHERE p.ProductID IN 
      (SELECT ProductID 
       FROM Production.ProductInventory AS pi 
       WHERE pi.Quantity = 0);