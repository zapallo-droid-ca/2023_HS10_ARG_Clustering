SELECT
	a.calendarCode, a.reporterCode, f.reporterCodeISO, f.reporterDesc, g.reporterRegionDesc, a.flowCode, d.flowDesc, a.partnerCode, b.partnerCodeISO, b.partnerDesc, c.partnerRegionDesc,  a.un_code_l2, e.desc_l2,
	netWeight, PrimaryValue
FROM
	(SELECT 
		calendarCode, reporterCode, flowCode, partnerCode, un_code_l2, SUM(netWeight) netWeight, SUM(PrimaryValue) PrimaryValue
	FROM dbo.ft_intComTrade
	WHERE un_freqCode = 'M' AND reporterCode = (SELECT unComtrade_id FROM dbo.dim_reporters WHERE CAST(unComTrade_text AS VARCHAR) = 'argentina') AND partnerCode <> '0' AND  un_code_l2 <> 'TOTAL'
	GROUP BY calendarCode, reporterCode, flowCode, partnerCode, un_code_l2) a 

	LEFT JOIN

	(SELECT unComtrade_id, unComtrade_text partnerDesc, alpha3ISO partnerCodeISO
	FROM dbo.dim_partners) b

	ON a.partnerCode = b.unComtrade_id

	LEFT JOIN

	(SELECT alpha3ISO, continent partnerRegionDesc
	FROM dbo.aux_countryContinent) c

	ON b.partnerCodeISO = c.alpha3ISO

	LEFT JOIN

	(SELECT un_flow_code, flow_desc flowDesc
	FROM dbo.dim_comTradeFlow) d

	ON a.flowCode = d.un_flow_code

	LEFT JOIN

	(SELECT un_code_l2, desc_l2
	FROM dbo.dim_HSclass_l2) e

	ON a.un_code_l2 = e.un_code_l2

	LEFT JOIN

	(SELECT unComtrade_id, unComtrade_text reporterDesc, alpha3ISO reporterCodeISO
	FROM dbo.dim_reporters) f

	ON a.reporterCode = f.unComtrade_id

	LEFT JOIN

	(SELECT alpha3ISO, continent reporterRegionDesc
	FROM dbo.aux_countryContinent) g

	ON f.reporterCodeISO = g.alpha3ISO

WHERE partnerCodeISO IS NOT NULL

