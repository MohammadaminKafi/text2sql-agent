SELECT p.ProductID,
       p.Name,
       p.ListPrice
FROM Production.Product AS p
WHERE p.ListPrice >
      (SELECT AVG(p2.ListPrice)
       FROM Production.Product AS p2
       WHERE p2.ProductSubcategoryID = p.ProductSubcategoryID);