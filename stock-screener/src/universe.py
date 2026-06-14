"""
universe.py
-----------
S&P 500 constituent list (as of June 2025).
Hardcoded to avoid scraping dependencies and rate limits.
"""

import logging
logger = logging.getLogger(__name__)


SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
    "ADI","ANSS","AON","APA","APH","AAPL","AMAT","APTV","ACGL","ADM","ANET",
    "AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL",
    "BAC","BK","BBWI","BAX","BDX","WRB","BBY","BIO","TECH","BIIB","BLK","BX",
    "BA","BCH","BMY","AVGO","BR","BRO","BF-B","BLDR","BXP","CHRW","CDNS","CZR",
    "CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW",
    "CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB",
    "CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
    "CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA",
    "CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY",
    "DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ",
    "DOV","DOW","DHR","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW",
    "EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR",
    "ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV",
    "FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FLT","FMC",
    "F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GS","HAL","HIG","HAS",
    "HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST",
    "HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR",
    "PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
    "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY",
    "KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS",
    "LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO",
    "MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH",
    "TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP",
    "NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH",
    "NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL",
    "OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG",
    "PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
    "PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG",
    "REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SO","LUV",
    "SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW",
    "TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO",
    "TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR","ULTA",
    "UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ",
    "VRTX","VFC","VTRS","VICI","V","VMC","WRK","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","GWW",
    "WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]


def get_sp500_tickers() -> list[str]:
    """Return the S&P 500 ticker list."""
    logger.info(f"Loaded {len(SP500_TICKERS)} S&P 500 tickers.")
    return SP500_TICKERS


def get_sample_tickers() -> list[str]:
    """
    A curated sample of 30 tickers across sectors for fast testing.
    Use this during development to avoid hitting rate limits.
    """
    return [
        # Technology
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        # Financials
        "JPM", "BAC", "GS", "V", "MA",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK",
        # Consumer
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Industrials
        "CAT", "DE", "BA", "HON", "UPS",
    ]