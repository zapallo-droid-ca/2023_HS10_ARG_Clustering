SELECT
            a.calendarCode_year AS 'year', b.alpha3ISO AS reporterISO, c.alpha3ISO AS partnerISO, b.unComtrade_text AS reporter, c.unComtrade_text AS partner ,a.flow , a.totalAnnualValue, a.totalValueHS10 totalCereals, a.totalValue1001 wheat, a.totalValue1002 rye, a.totalValue1003 barley, a.totalValue1004 oats, a.totalValue1005 maize, a.totalValue1006 rice, a.totalValue1007 sorghum, a.totalValue1008 others
            FROM
            	(SELECT 
            	calendarCode_year ,reporterCode, CASE flowCode WHEN 'X' THEN 'Export' WHEN 'M' THEN 'Import' END flow,
            	partnerCode, totalAnnualValue, totalValueHS10, totalValue1001,totalValue1002,totalValue1003,totalValue1004 ,totalValue1005 ,totalValue1006, totalValue1007, totalValue1008
            	FROM iDevelopStg.dbo.ft_inComTradeHS10_Annual) a
            
            	LEFT JOIN 
            
            	(SELECT unComtrade_id, unComtrade_text, alpha3ISO
            	FROM iDevelopStg.dbo.dim_reporters) b
            
            	ON a.reporterCode = b.unComtrade_id
            
            	LEFT JOIN
            
            	(SELECT unComtrade_id, unComtrade_text, alpha3ISO
            	FROM iDevelopStg.dbo.dim_partners) c
            
            	ON a.partnerCode = c.unComtrade_id
            
            	SELECT * FROM dim_HSclass_l2 WHERE un_code_l2 LIKE '100%'