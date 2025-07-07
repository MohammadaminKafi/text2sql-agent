SELECT
  ea.BusinessEntityID,
  LOWER(SUBSTRING(ea.EmailAddress, 1, CHARINDEX('@', ea.EmailAddress) - 1)) AS Username,
  UPPER(SUBSTRING(ea.EmailAddress, CHARINDEX('@', ea.EmailAddress) + 1, LEN(ea.EmailAddress))) AS Domain,
  LEN(ea.EmailAddress) AS EmailLength
FROM Person.EmailAddress AS ea
WHERE ea.EmailAddress LIKE '%@%.%'
ORDER BY EmailLength DESC;