SELECT p.Name,
       p.ListPrice,
       (SELECT AVG(ListPrice) 
        FROM Production.Product 
        WHERE ProductSubcategoryID = p.ProductSubcategoryID) AS AvgPriceInSubcategory,
       (SELECT MAX(ListPrice) 
        FROM Production.Product 
        WHERE ProductSubcategoryID = p.ProductSubcategoryID) AS MaxPriceInSubcategory
FROM Production.Product AS p
WHERE p.ListPrice > 
      (SELECT AVG(ListPrice) 
       FROM Production.Product 
       WHERE ProductSubcategoryID = p.ProductSubcategoryID);