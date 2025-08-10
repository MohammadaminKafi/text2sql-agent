SELECT 
	pg.ProductCategoryID AS CategoryID,
	pg.[Name] AS CategoryName,
	MAX(p.ListPrice) AS MaxListPrice,
	AVG(p.ListPrice) AS AvgListPrice
FROM
	Production.[Product] p
	JOIN Production.ProductSubcategory psg ON p.ProductSubcategoryID = psg.ProductSubcategoryID
	JOIN Production.ProductCategory pg ON psg.ProductCategoryID = pg.ProductCategoryID
GROUP BY
	pg.[Name],
	pg.ProductCategoryID
ORDER BY
	AvgListPrice;