SELECT sp.StateProvinceCode,
       COUNT(*) AS AddressCount
FROM Person.Address       AS a
JOIN Person.StateProvince AS sp ON sp.StateProvinceID = a.StateProvinceID
GROUP BY sp.StateProvinceCode
HAVING COUNT(*) >
      (SELECT AVG(StateCount)
       FROM (SELECT COUNT(*) AS StateCount
             FROM Person.Address
             GROUP BY StateProvinceID) AS s);