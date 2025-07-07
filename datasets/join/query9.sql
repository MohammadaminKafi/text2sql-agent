SELECT p.Name AS ProductName, pv.StandardPrice, v.Name AS VendorName, v.CreditRating, at.Name AS ProductType
FROM Production.Product AS p
JOIN Purchasing.ProductVendor AS pv ON p.ProductID = pv.ProductID
JOIN Purchasing.Vendor AS v ON pv.BusinessEntityID = v.BusinessEntityID
JOIN Production.ProductSubcategory AS ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID
JOIN Production.ProductCategory AS pc ON ps.ProductCategoryID = pc.ProductCategoryID
JOIN Production.ProductModel AS pm ON p.ProductModelID = pm.ProductModelID
JOIN Production.Illustration AS il ON pm.IllustrationID = il.IllustrationID
JOIN Production.ProductModelIllustration AS pmi ON pm.ProductModelID = pmi.ProductModelID
JOIN Production.UnitMeasure AS um ON p.SizeUnitMeasureCode = um.UnitMeasureCode
JOIN Production.ProductDescription AS pd ON pm.ProductModelID = pd.ProductModelID
JOIN Production.ProductModelProductDescriptionCulture AS pdpc ON pd.ProductDescriptionID = pdpc.ProductDescriptionID
JOIN Production.Culture AS at ON pdpc.CultureID = at.CultureID
WHERE at.CultureID = 'en';