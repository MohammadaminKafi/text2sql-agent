SELECT pc.ProductCategoryID,
       pc.Name,
       COUNT(p.ProductID) AS ProductCount
FROM Production.ProductCategory    AS pc
JOIN Production.ProductSubcategory AS psc ON psc.ProductCategoryID = pc.ProductCategoryID
JOIN Production.Product            AS p   ON p.ProductSubcategoryID = psc.ProductSubcategoryID
GROUP BY pc.ProductCategoryID, pc.Name
HAVING COUNT(p.ProductID) >
      (SELECT AVG(CategoryCount)
       FROM (SELECT COUNT(*) AS CategoryCount
             FROM Production.ProductCategory        AS pc2
             JOIN Production.ProductSubcategory     AS psc2 ON psc2.ProductCategoryID = pc2.ProductCategoryID
             JOIN Production.Product                AS p2   ON p2.ProductSubcategoryID = psc2.ProductSubcategoryID
             GROUP BY pc2.ProductCategoryID) AS counts);