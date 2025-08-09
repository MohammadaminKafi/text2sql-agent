SELECT 
    st.Name AS Territory,
    COUNT(DISTINCT soh.CustomerID) AS CustCount,
    SUM(soh.Freight) AS TotalFreight,
    AVG(soh.Freight) AS AvgFreight
FROM Sales.SalesOrderHeader AS soh
JOIN Sales.SalesTerritory AS st
  ON soh.TerritoryID = st.TerritoryID
GROUP BY st.Name
ORDER BY TotalFreight DESC;