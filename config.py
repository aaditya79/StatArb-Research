from dataclasses import dataclass, field
from typing import Dict, List, Optional


SECTOR_ETFS: List[str] = [
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY",
    "IYR", "IYT", "OIH", "SMH", "RTH", "RKH", "UTH",
    "XLB",
    "XLU",
    "GDX",
    "IBB",
    "KRE",
    "XOP",
    "XHB",
    "XRT",
    "IYZ",
    "XME",
]

MARKET_ETF: str = "SPY"

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS", "WFC", "C",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "UNH", "ABT", "MRK",
    "HD", "PG", "KO", "PEP", "WMT",
    "NEE", "DUK", "SO", "D", "AEP",
    "UNP", "UPS", "FDX", "CSX", "NSC",
    "NVDA", "INTC", "AVGO", "TXN", "QCOM",
]

# ─────────────────────────────────────────────────────────────────────────────
# Paper-style universe (~1,000 names)
#
# Avellaneda & Lee (2010) use the point-in-time CRSP universe of roughly
# 1,417 US stocks with market cap > $1B on each trading date, 1997-2007.
# We cannot reconstruct that exactly without per-date constituent lists, but
# the list below is a close static proxy: S&P 500 survivors + S&P 400/600
# large-caps + major delisted names (Lehman, Bear Stearns, Fannie, Freddie,
# Compaq, Sun, WaMu, Wyeth, Schering-Plough, Burlington Northern, XTO,
# pre-merger airlines, pre-2005 Kraft, etc.) that were actively traded
# during the 1997-2007 sample.
#
# To use it, swap this block in for DEFAULT_TICKERS above:
#     DEFAULT_TICKERS = PAPER_TICKERS
#
# Data-quality notes (already handled in the pipeline):
#   - Delisted names: the CRSP data source maps them via historical PERMNO
#     lookups. With yfinance many of the delisted tickers return nothing;
#     those names are silently dropped by the `available` filter in
#     app/Home.py. Prefer CRSP for this universe.
#   - IPOs mid-sample: returns are NaN before the first trading day; the
#     OU fitter ignores those rows via the np.isfinite mask.
#   - PCA window with lots of IPO/delisting: the helper fills isolated NaN
#     with 0 (unbiased for log-returns) so a single missing print doesn't
#     collapse the 252-day window (fix in statarb/factors/pca.py).
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TICKERS: List[str] = [
    # --- Technology / Hardware / Semis (XLK, SMH) ---
    "AAPL", "MSFT", "ORCL", "IBM", "HPQ", "DELL", "CSCO", "INTC", "AMD",
    "NVDA", "TXN", "QCOM", "MU", "AMAT", "KLAC", "LRCX", "ADI",
    "ADSK", "CTXS", "CA", "SYMC", "INTU", "CRM", "ADBE", "BMC", "CHKP",
    "VRSN", "AKAM", "EBAY", "YHOO", "GOOG", "GOOGL", "AMZN", "PCLN", "EXPE",
    "NTAP", "EMC", "STX", "WDC", "SNDK", "BRCD", "JNPR", "FFIV", "RHT",
    "TLAB", "CIEN", "FNSR", "JDSU", "NVLS", "LLTC", "MCHP", "XLNX", "ALTR",
    "BRCM", "MRVL", "NVLSQ", "ATML", "FSL", "ONNN", "TER", "CY", "IRF",
    "POWI", "SWKS", "RFMD", "TQNT", "AVCT", "PAYX", "ADP", "FIS",
    "FISV", "JKHY", "WU", "MA", "DFS", "COF", "AXP", "GLW", "CPQ",
    "SUNW", "NCR", "PALM", "RIMM", "MOT", "LU", "NT",
    "SCMR", "LSI", "AGR",

    # --- Financials: Banks / Broker-Dealers / Asset Mgmt / Insurance (XLF, KRE) ---
    "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "BK", "STT",
    "NTRS", "SCHW", "TROW", "BEN", "IVZ", "AMG", "JNS", "AFL", "AIG",
    "ALL", "HIG", "MET", "PRU", "TRV", "CB", "CINF", "PGR", "XL", "LNC",
    "MMC", "AJG", "AON", "WRB", "AFG", "RNR", "MKL", "RE", "AXS", "BRO",
    "CNA", "HCC", "ORI", "L", "UNM", "GNW", "STAN", "WM",  # WaMu delisted
    "LEH",  # Lehman (delisted 2008)
    "BSC",  # Bear Stearns (acquired 2008)
    "MER",  # Merrill (acquired 2009)
    "ABK", "MBI", "AFC", "AMTD", "ETFC", "LM", "EV", "FITB", "KEY",
    "HBAN", "MTB", "RF", "CMA", "ZION", "BBT", "STI", "SNV", "FHN", "PBCT",
    "NYB", "IFC", "WL", "CBH", "MI",  # Marshall & Ilsley (delisted)
    "CFC",  # Countrywide (delisted 2008)
    "NCC",  # National City (delisted 2008)
    "FNM",  # Fannie Mae (delisted 2008 → OTC)
    "FRE",  # Freddie Mac (delisted 2008 → OTC)

    # --- Energy: Majors / E&P / Services (XLE, XOP, OIH) ---
    "XOM", "CVX", "COP", "OXY", "MRO", "HES", "APA", "DVN", "APC", "EOG",
    "CHK", "XTO",  # XTO delisted 2010
    "PXD", "CAM", "NBL", "RRC", "STR", "WLL", "CLR", "COG", "QEP", "SWN",
    "EQT", "BEXP", "CRZO", "MUR", "DNR", "CXO", "OAS",
    "SLB", "HAL", "BHI", "NOV", "FTI", "PDE",  # Pride Intl
    "RIG", "ESV", "NE", "ATW", "RDC", "DO", "PTEN", "NBR", "BJS",  # BJ Services delisted
    "SII",  # Smith Intl delisted 2010
    "WFT", "HERO", "CFW", "PKD", "HP", "UNT", "VLO", "TSO",
    "SUN",  # Sunoco-legacy
    "HOC", "FTO",  # Frontier Oil delisted
    "WNR", "TGP", "WMB", "EPD", "EP",  # El Paso delisted 2012

    # --- Healthcare / Pharma / Biotech / Med Devices (XLV, IBB) ---
    "JNJ", "PFE", "MRK", "BMY", "ABT", "LLY", "WYE",  # Wyeth delisted 2009 (→PFE)
    "SGP",  # Schering-Plough delisted 2009 (→MRK)
    "FRX",  # Forest Labs delisted 2014
    "MYL", "WPI", "PRGO", "ENDP", "AGN", "ALXN", "BIIB", "CELG",
    "AMGN", "GILD", "VRTX", "REGN", "ALKS", "INCY", "JAZZ", "MDCO",
    "ILMN", "LIFE", "AFFX", "SIAL", "PKI", "MTD", "WAT", "BIO", "TECH",
    "BMET",  # Biomet delisted 2007
    "MDT", "SYK", "BSX", "ZMH", "STJ", "BDX", "BAX", "CAH", "MCK", "ABC",
    "HSIC", "PDCO", "CERN", "MDRX", "QSII", "CYH", "HCA", "THC", "UHS",
    "LPNT", "HMA", "HLS",  # HealthSouth
    "UNH", "CI", "AET", "HUM", "WLP",  # WellPoint (became ANTM)
    "HNT",  # Health Net
    "MOH", "CNC", "GTS", "DHR", "COV", "ISRG", "VAR", "HOLX",
    "RMD", "ABMD", "XRAY", "RGEN", "EW",

    # --- Consumer Discretionary: Retail / Autos / Media (XLY, XRT, RTH, XHB) ---
    "HD", "LOW", "TGT", "COST", "KR", "SWY",  # Safeway delisted 2015
    "BBY", "SPLS", "ODP", "OMX",  # OfficeMax delisted 2013
    "TJX", "ROST", "KSS", "JCP", "M", "JWN", "DDS", "BIG", "DLTR", "FDO",
    "BJ",  # BJ's Wholesale delisted 2011
    "RSH", "ANF", "LTD", "GPS", "URBN", "AEO", "TLB",  # Talbots delisted 2012
    "CHS", "CATO", "ZUMZ", "PSUN",  # Pacific Sunwear
    "FL", "PLCE", "WTSLA", "DEST", "CWTR", "BEBE", "DLIA",
    "SBUX", "MCD", "YUM", "CMG", "DRI", "EAT", "BJRI", "DIN", "PNRA",
    "TXRH", "BWLD", "DPZ", "SONC", "RT",  # Ruby Tuesday
    "F", "GM", "HOG", "JCI", "LEA", "BWA", "DLPH", "TEN", "ALV", "TRW",
    "VC", "TKR", "SNA", "WHR", "LEG", "MHK", "NVR", "LEN", "DHI", "PHM",
    "TOL", "KBH", "MDC", "RYL", "MTH", "BZH", "HOV", "WCI", "SPF",
    "DIS", "CMCSA", "CMCSK", "VIA", "VIAB", "TWX",  # Time Warner (became WBD)
    "CBS", "NWS", "NWSA", "DISH", "DTV",  # DirecTV delisted 2015
    "SIRI", "XMSR", "TRIP", "LVS", "WYNN", "MGM", "HET",
    "ISLE", "CZR", "PENN", "BYI", "WMS", "GTK", "SGMS", "IGT",

    # --- Consumer Staples (XLP) ---
    "PG", "KO", "PEP", "KMB", "CL", "CHD", "CLX", "EL", "AVP",
    "MO", "RAI",  # Reynolds
    "STZ", "TAP", "BFB",
    "GIS", "K", "CPB", "CAG", "MKC", "HRL", "SJM", "TSN", "ADM", "BG",
    "KFT",  # Kraft pre-split 2012
    "SLE",  # Sara Lee delisted 2012
    "WAG", "CVS", "RAD", "WFM", "SVU", "DLM", "HNZ",  # Heinz delisted 2013
    "DF", "POST", "FLO", "ENR", "NWL",

    # --- Industrials / Defense / Airlines / Rails / Machinery (XLI, IYT) ---
    "BA", "GE", "HON", "MMM", "UTX", "LMT", "RTN", "GD", "NOC", "COL",
    "LLL", "TXT", "PCP", "HEI", "TDG", "GY",
    "CAT", "DE", "CMI", "PCAR", "NAV", "PH", "ETN", "EMR", "ROK", "ROP",
    "DOV", "ITT", "HUBB", "AME", "APH", "ITW", "TYC",
    "LUK", "FLS", "PNR", "WTS", "LII", "AOS", "MLM", "VMC", "EXP", "SUM",
    "UNP", "CSX", "NSC", "BNI",  # BNSF (delisted 2010, acquired by BRK)
    "KSU", "CP", "CNI", "GWR", "JBHT", "LSTR", "ODFL", "CHRW", "EXPD",
    "FDX", "UPS",
    "AMR",  # pre-merger American Airlines (delisted 2013)
    "UAUA",  # pre-merger UAL (delisted 2010)
    "CAL",  # pre-merger Continental (delisted 2010)
    "DAL", "NWAC",  # Northwest Airlines
    "LCC",  # US Airways (delisted 2013)
    "LUV", "JBLU", "ALK", "SKYW", "R", "DSX", "EXM", "OSG", "TNK",
    "GNK", "DRYS", "EGLE", "BALT", "NM", "FRO",

    # --- Utilities (XLU, UTH) ---
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "PPL", "ETR", "PCG",
    "ED", "SRE", "PEG", "EIX", "FE", "AEE", "CMS", "CNP", "DTE", "NI",
    "NRG", "WEC", "SCG", "PNW", "POM", "VVC", "ALE", "IDA", "MGEE", "OGE",
    "PNM", "WR", "BKH", "HE", "ITC", "UIL", "UNS", "AVA", "WGL", "LG",
    "SJI", "NJR", "NWN", "PNY", "SWX", "NSTR", "CPN",

    # --- Materials / Metals / Chemicals (XLB, XME, GDX) ---
    "DD", "DOW", "MON", "POT", "MOS", "CF", "AGU", "IPI", "SYT", "FMC",
    "EMN", "IFF", "PPG", "SHW", "RPM", "CBT", "ALB", "FUL", "LYB", "WLK",
    "HUN", "CC", "OLN", "ARG", "APD", "PX", "AA",  # Alcoa pre-split
    "CENX", "KALU", "X", "NUE", "STLD", "ATI", "CRS", "CMC", "HAYN",
    "ROCK", "HSC", "AKS", "WOR", "BLL", "IP", "WY", "PKG", "BZ",
    "SON", "GEF", "LPX", "BCC", "DEL", "CDE", "NEM", "ABX", "GG", "AUY",
    "KGC", "IAG", "EGO", "HMY", "FCX", "TC", "PAAS", "SSRI", "AG", "HL",
    "SLW", "RGLD",

    # --- Real Estate REITs (IYR) ---
    "SPG", "PSA", "AVB", "EQR", "BXP", "VNO", "HCN", "HCP", "VTR", "O",
    "NNN", "REG", "SLG", "DRE", "FRT", "KIM", "DDR", "GGP", "PEI",
    "MAC", "TCO", "ESS", "CPT", "UDR", "AIV", "EQY", "RYN", "PLD", "AMB",
    "DLR", "MAA", "EXR", "PSB", "CUZ", "HIW", "BRE", "AHT", "LHO", "HST",
    "DRH", "SHO", "BEE", "FCH", "IHR", "FR",

    # --- Telecom / Communications (IYZ) ---
    "T", "VZ", "CTL", "S",  # Sprint
    "BLS",  # BellSouth (delisted 2006, acquired by AT&T)
    "SBC",  # SBC Communications (became T in 2005)
    "Q",  # Qwest (delisted 2011, acquired by CTL)
    "FTR", "WIN", "TDS", "USM", "LVLT", "CNSL", "GNCMA", "ALSK",
    "CTB", "CBB", "PAET", "IPG", "OMC", "MDP", "WPO", "NYT", "GCI",
    "LEE", "JRN", "MNI",

    # --- Misc large-caps, conglomerates ---
    "GWW", "FAST", "MSM",
    "HRS", "HAR", "GRMN", "TRMB", "CGNX", "ROG", "IEX",
]
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = PAPER_TICKERS

# ─────────────────────────────────────────────────────────────────────────────
# Modern-era universe (~2014-2025, ~400 names)
#
# Curated for backtests starting roughly 2014. Includes:
#   - Post-2010 IPOs that are dominant in modern returns (META, TSLA-era
#     mega-caps, SaaS/cloud, fintech, cannabis, EV, crypto-adjacent)
#   - Traditional S&P 500/400 large-caps still trading today
#   - Excludes the pre-2014 delistings that poison PAPER_TICKERS on modern
#     windows (LEH, BSC, MER, CPQ, SUNW, SBC, BLS, KFT-pre-split, SLE, etc.)
#
# To use it, swap this block in for DEFAULT_TICKERS:
#     DEFAULT_TICKERS = MODERN_TICKERS
#
# Caveats:
#   - The mean-reversion anomaly has materially decayed post-2010 (paper's
#     own Table 5 shows Sharpe turning negative by 2005-2007). Expect
#     weak-to-negative Sharpe on this universe — that is the empirical
#     finding, not a code bug.
#   - Recent IPOs (ABNB, COIN, DASH, RBLX, HOOD, AFRM, UPST, PLTR, SNOW,
#     RIVN, LCID, etc.) have only a few years of history; the 252-day PCA
#     window + 60-day OU window need ~300 days before they contribute signal.
# ─────────────────────────────────────────────────────────────────────────────
MODERN_TICKERS: List[str] = [
    # --- Mega-cap tech ---
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",

    # --- Semis ---
    "AVGO", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "KLAC", "LRCX",
    "ADI", "MRVL", "MCHP", "MPWR", "NXPI", "ON", "SWKS", "QRVO", "TER",
    "ENTG", "TSM", "ASML", "STM", "ARM",

    # --- Enterprise software / SaaS / cloud ---
    "ORCL", "ADBE", "CRM", "NOW", "INTU", "WDAY", "TEAM", "ADSK", "ANSS",
    "SNPS", "CDNS", "ZS", "OKTA", "CRWD", "PANW", "FTNT", "NET", "DDOG",
    "MDB", "SNOW", "PLTR", "ESTC", "ZI", "BILL", "HUBS", "MNDY", "GTLB",
    "DBX", "BOX", "SMAR", "CFLT", "S", "VEEV", "TYL", "APPN", "ASAN",
    "PATH", "AI", "FROG", "PCTY", "PAYC", "PD", "SPLK", "RPD", "CYBR",
    "TENB", "QLYS", "VRNS", "SAIL", "NTCT",

    # --- Consumer internet / media / streaming / gaming ---
    "NFLX", "DIS", "CMCSA", "WBD", "PARA", "FOXA", "FOX", "SPOT", "ROKU",
    "EA", "TTWO", "ATVI",  # ATVI acquired 2023
    "ZG", "Z", "TRIP", "BKNG", "EXPE", "ABNB", "UBER", "LYFT", "DASH",
    "SNAP", "PINS", "MTCH", "ETSY", "EBAY", "CHWY", "W", "RBLX", "U",
    "TTD", "DV", "IAS", "LYV", "SIRI",

    # --- E-commerce / marketplaces (incl. intl ADRs) ---
    "SHOP", "MELI", "SE", "BABA", "JD", "PDD", "BIDU", "NTES", "TME",

    # --- Fintech / payments / exchanges ---
    "V", "MA", "PYPL", "SQ",  # Block
    "AXP", "DFS", "COF", "FIS", "FISV",  # Fiserv
    "JKHY", "WU", "GPN", "ADP", "PAYX", "NDAQ", "ICE", "CME", "CBOE",
    "MSCI", "SPGI", "MCO", "MKTX", "TW", "COIN", "HOOD", "SOFI", "AFRM",
    "UPST", "LC", "MQ", "PGY", "DAVE", "FOUR",

    # --- Banks / broker-dealers / asset mgmt ---
    "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "BK", "STT",
    "NTRS", "SCHW", "TROW", "BEN", "IVZ", "BLK", "AMG", "CG", "KKR",
    "APO", "BX", "ARES", "OWL", "JEF", "LPLA", "RJF", "FITB", "KEY",
    "HBAN", "MTB", "RF", "CMA", "ZION", "CFG", "TFC",  # Truist
    "FHN", "WAL", "SNV", "PNFP", "CFR", "OZK", "EWBC", "HTH",

    # --- Insurance ---
    "AIG", "ALL", "HIG", "MET", "PRU", "TRV", "CB",
    "CINF", "PGR", "LNC", "UNM", "RGA", "MMC", "AJG", "AON",
    "WRB", "AFG", "RNR", "MKL", "RE", "AXS", "BRO", "CNA", "L", "AFL",

    # --- Healthcare services / managed care / pharmacy / distribution ---
    "UNH", "CI", "HUM", "ELV",  # Anthem → Elevance
    "CVS", "WBA", "MCK", "CAH", "ABC", "COR",
    "HSIC", "PDCO", "HCA", "THC", "UHS", "CYH", "DVA",
    "LH", "DGX", "CRL", "IQV", "ICLR", "CTLT",

    # --- Big pharma ---
    "JNJ", "PFE", "MRK", "BMY", "ABT", "LLY", "ABBV", "AZN", "GSK",
    "NVS", "SNY", "NVO", "TEVA", "VTRS", "ENDP", "PRGO",

    # --- Biotech ---
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "BMRN", "INCY", "JAZZ",
    "EXAS", "NBIX", "NVAX", "MRNA", "BNTX", "SRPT", "RARE", "BLUE",
    "FOLD", "IONS", "CRSP", "EDIT", "NTLA", "BEAM", "VERV", "ARVN",
    "INSM", "ACAD", "HALO", "LGND", "ILMN", "TDOC", "DXCM", "PODD",

    # --- Medical devices / diagnostics ---
    "MDT", "SYK", "BSX", "ZBH", "BDX", "BAX", "ISRG", "EW", "HOLX",
    "RMD", "XRAY", "ALGN", "TNDM", "INSP", "PEN", "NVRO", "NARI",
    "IRTC", "IART", "GMED", "NVST", "DHR", "TMO", "A", "WAT", "MTD",
    "PKI", "BIO", "TECH", "TFX",

    # --- Retail: mass / discount / home / specialty ---
    "WMT", "COST", "TGT", "HD", "LOW", "KR", "DG", "DLTR", "FIVE",
    "BBY", "ULTA", "TJX", "ROST", "BURL", "KSS", "JWN", "M", "DDS",
    "BIG", "OLLI",
    "LULU", "NKE", "UA", "UAA", "DECK", "CROX", "COLM", "VFC",
    "PVH", "RL", "TPR", "CPRI",
    "AAP", "AZO", "ORLY", "LKQ", "POOL",

    # --- Restaurants / staples / beverage / tobacco ---
    "MCD", "SBUX", "YUM", "CMG", "DRI", "QSR", "WEN", "JACK", "TXRH",
    "SHAK", "DPZ", "PZZA", "CAKE", "DNUT", "BROS", "EAT",
    "PG", "KO", "PEP", "KHC", "MDLZ", "KMB", "CL", "CHD", "CLX",
    "EL", "COTY", "NWL",
    "MO", "PM", "BTI", "STZ", "DEO", "TAP", "BUD", "SAM", "BFB",
    "MNST", "KDP", "CELH",
    "GIS", "K", "CPB", "CAG", "MKC", "HRL", "SJM", "TSN", "ADM", "BG",
    "POST", "FLO", "LW",

    # --- Autos / EV / ride-hail / rentals ---
    "F", "GM", "STLA", "RIVN", "LCID", "NIO", "XPEV", "LI",
    "FSR", "NKLA", "PSNY", "HTZ", "CAR",
    "HOG", "PII", "BC", "BWA", "LEA", "DAN", "MGA", "ALV", "APTV",
    "VC", "SNA", "GPC",

    # --- Industrials / machinery / aerospace / defense ---
    "BA", "GE", "HON", "MMM", "LMT", "RTX", "GD", "NOC", "LHX",
    "TDG", "HEI", "BWXT", "HII", "KTOS", "AXON", "PH", "EMR", "ROK",
    "ROP", "DOV", "ITT", "HUBB", "AME", "APH", "ITW", "CAT", "DE",
    "CMI", "PCAR", "PNR", "WTS", "LII", "AOS", "FLS", "FAST", "GWW",
    "MSM", "WSO", "WCC",

    # --- Rails / airlines / logistics / shipping ---
    "UNP", "CSX", "NSC", "CP", "CNI",
    "FDX", "UPS", "CHRW", "EXPD", "JBHT", "LSTR", "ODFL", "SAIA",
    "XPO", "GXO", "ARCB", "MATX", "SBLK", "GOGL", "ZIM", "FRO",
    "DAL", "UAL", "AAL", "LUV", "JBLU", "ALK", "SKYW", "HA", "SAVE",
    "ALGT", "ULCC",

    # --- Energy: majors / E&P / services ---
    "XOM", "CVX", "COP", "OXY", "HES", "EOG", "PXD", "DVN", "APA",
    "MRO", "FANG", "PR", "MUR", "SM", "OVV", "CHRD", "MTDR", "CIVI",
    "CRC", "CRGY", "NOG", "CNX", "EQT", "AR", "RRC", "SWN",
    "VLO", "MPC", "PSX", "DK", "PBF", "DINO", "CVI",
    "SLB", "HAL", "BKR", "NOV", "FTI", "CHX", "TS", "WFRD",
    "LBRT", "NBR", "PTEN", "PUMP", "RIG", "NE", "VAL", "DO", "BORR",

    # --- Midstream / pipelines / LNG ---
    "WMB", "KMI", "EPD", "ET", "MPLX", "OKE", "TRGP", "PAA", "ENB",
    "TRP", "LNG", "CQP", "NFG", "DTM", "WES",

    # --- Utilities ---
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "PPL", "ETR", "PCG",
    "ED", "SRE", "PEG", "EIX", "FE", "AEE", "CMS", "CNP", "DTE", "NI",
    "NRG", "WEC", "PNW", "POR", "EVRG", "VST", "CEG", "TLN",
    "BEP", "BEPC", "NEP", "AES", "ATO", "ES", "LNT",

    # --- Materials / chemicals / metals / mining ---
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "RPM", "FMC",
    "EMN", "IFF", "ALB", "CE", "OLN", "ASH", "LYB", "WLK",
    "CF", "MOS", "NTR",
    "FCX", "NEM", "GOLD", "AEM", "KGC", "AGI", "PAAS", "WPM", "FNV",
    "NUE", "STLD", "CLF", "X", "MT", "RS", "CMC", "ATI", "CRS",
    "AA", "CENX", "KALU", "TROX", "TSE",
    "VMC", "MLM", "EXP", "SUM", "IP", "WY", "PKG", "BALL",
    "SON", "GEF", "LPX", "AVY", "BERY", "SEE", "SLVM",

    # --- REITs / real estate ---
    "PLD", "AMT", "CCI", "EQIX", "DLR", "SPG", "PSA", "WELL", "AVB",
    "EQR", "VTR", "O", "NNN", "REG", "SLG", "VNO", "BXP", "FRT", "KIM",
    "MAC", "HST", "PK", "DRH", "SHO", "XHR", "RHP", "APLE",
    "ARE", "ESS", "CPT", "UDR", "MAA", "EXR", "CUBE", "LSI", "HR",
    "VICI", "GLPI", "IRM", "RYN", "PCH", "CTRE", "SBRA", "OHI",

    # --- Telecom / communication services ---
    "VZ", "T", "TMUS", "CHTR", "LUMN", "FYBR", "USM", "TDS", "IRDM",

    # --- Cannabis ---
    "CGC", "TLRY", "ACB", "CRON", "GRWG", "SNDL", "OGI",

    # --- Crypto / miners / exchanges ---
    "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "IREN",
    "WULF", "HIVE",

    # --- Misc / industrial tech ---
    "IEX", "GRMN", "TRMB", "CGNX",
]
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_TICKERS = MODERN_TICKERS

SECTOR_TO_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Real Estate": "IYR",
    "Utilities": "UTH",
    "Basic Materials": "XLI",
    "Communication Services": "XLK",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Materials": "XLB",
    "Semiconductors": "SMH",
    "Telecom": "IYZ",
    "Telecommunications": "IYZ",
}

# ─────────────────────────────────────────────────────────────────────────────
# TICKER_TO_ETF_OVERRIDES: hard-coded sector assignments, checked BEFORE any
# data-source lookup. Used for:
#   (a) delisted tickers whose yfinance `.info` returns nothing and whose
#       CRSP SIC code may be missing or misleading,
#   (b) ticker-symbol collisions where the current holder of the symbol is
#       in a different sector than the historical paper-era company (WM,
#       SUN, STR, Q, SBC, BLS — see PAPER_TICKERS report).
#
# Coverage is focused on names in PAPER_TICKERS that are hard to resolve
# from a live data provider. Anything not in this dict falls through to
# the data source's sector lookup; whatever's still unresolved gets XLY.
# ─────────────────────────────────────────────────────────────────────────────
TICKER_TO_ETF_OVERRIDES: Dict[str, str] = {
    # Financials (banks / broker-dealers / insurance delistings)
    "LEH": "XLF", "BSC": "XLF", "MER": "XLF", "CFC": "XLF", "NCC": "XLF",
    "FNM": "XLF", "FRE": "XLF", "WM": "XLF",  # WaMu — collides with Waste Mgmt on yfinance
    "ABK": "XLF", "MBI": "XLF", "MI": "XLF", "IFC": "XLF", "WL": "XLF",
    "CBH": "XLF", "NYB": "XLF", "BBT": "XLF", "STI": "XLF", "PBCT": "XLF",
    "STAN": "XLF", "AMTD": "XLF", "ETFC": "XLF", "LM": "XLF", "EV": "XLF",
    "AFC": "XLF", "JNS": "XLF", "SCHW": "XLF", "GNW": "XLF",

    # Tech (hardware / semis / software / internet delistings)
    "CPQ": "XLK", "SUNW": "XLK", "YHOO": "XLK", "EMC": "XLK", "SNDK": "SMH",
    "BRCD": "XLK", "RHT": "XLK", "TLAB": "XLK", "CIEN": "XLK", "FNSR": "XLK",
    "JDSU": "XLK", "NVLS": "SMH", "NVLSQ": "SMH", "LLTC": "SMH", "ALTR": "SMH",
    "BRCM": "SMH", "ATML": "SMH", "FSL": "SMH", "ONNN": "SMH", "CY": "SMH",
    "IRF": "SMH", "TQNT": "SMH", "SCMR": "XLK", "RFMD": "SMH", "PALM": "XLK",
    "RIMM": "XLK", "MOT": "XLK", "LU": "XLK", "NT": "XLK", "AGR": "SMH",
    "LSI": "SMH", "CA": "XLK", "SYMC": "XLK", "BMC": "XLK", "AVCT": "XLK",
    "PCLN": "XLY",  # Priceline (→BKNG) — consumer travel, not tech
    "VRSN": "XLK", "CHKP": "XLK", "CTXS": "XLK", "NTAP": "XLK", "EBAY": "XLY",
    "AKAM": "XLK", "FFIV": "XLK", "JNPR": "XLK", "MRVL": "SMH", "MCHP": "SMH",
    "XLNX": "SMH", "SWKS": "SMH", "POWI": "SMH", "TER": "SMH", "NCR": "XLK",

    # Healthcare delistings (pharma / biotech / devices)
    "WYE": "XLV", "SGP": "XLV", "FRX": "XLV", "MYL": "XLV", "WPI": "XLV",
    "AGN": "XLV", "ALXN": "IBB", "CELG": "IBB", "MDCO": "IBB", "LIFE": "IBB",
    "AFFX": "IBB", "SIAL": "XLV", "BMET": "XLV", "HLS": "XLV", "LPNT": "XLV",
    "HMA": "XLV", "WLP": "XLV", "HNT": "XLV", "CERN": "XLV", "MDRX": "XLV",
    "QSII": "XLV", "COV": "XLV",

    # Energy delistings
    "XTO": "XLE", "APC": "XLE", "PXD": "XLE", "NBL": "XLE", "CAM": "OIH",
    "STR": "XLU",  # Questar (natural-gas utility) — collides with Sitio Royalties on yfinance
    "BHI": "OIH", "PDE": "OIH", "BJS": "OIH", "SII": "OIH", "WFT": "OIH",
    "HERO": "OIH", "CFW": "OIH", "PKD": "OIH", "UNT": "XLE", "TSO": "XLE",
    "HOC": "XLE", "FTO": "XLE", "WNR": "XLE", "EP": "XLE", "CHK": "XLE",
    "BEXP": "XLE", "CRZO": "XLE", "CXO": "XLE", "DNR": "XLE", "OAS": "XLE",
    "QEP": "XLE", "MUR": "XLE", "SWN": "XLE", "RDC": "OIH", "ATW": "OIH",
    "ESV": "OIH", "TGP": "XLE",
    "SUN": "XLE",  # legacy Sunoco — collides with SunCoke on yfinance

    # Telecom delistings
    "BLS": "IYZ",  # BellSouth — collides with ... nothing current but no yfinance data
    "SBC": "IYZ",  # SBC Communications
    "Q": "IYZ",    # Qwest
    "AWE": "IYZ",

    # Consumer Discretionary delistings
    "TWX": "XLY", "VIA": "XLY", "VIAB": "XLY", "CBS": "XLY", "DTV": "XLY",
    "XMSR": "XLY", "HET": "XLY", "ISLE": "XLY", "PNRA": "XLY", "BWLD": "XLY",
    "RT": "XLY", "BJ": "XLY", "TLB": "XLY", "PSUN": "XLY", "SPLS": "XLY",
    "OMX": "XLY", "FDO": "XLY", "WTSLA": "XLY", "DEST": "XLY", "CWTR": "XLY",
    "BEBE": "XLY", "DLIA": "XLY", "DYN": "XLY", "SWY": "XLP",
    "WYN": "XLY", "HOT": "XLY", "MAR": "XLY", "GTK": "XLY", "SGMS": "XLY",
    "WMS": "XLY", "BYI": "XLY", "CMCSK": "XLY", "NWS": "XLY", "NWSA": "XLY",
    "FOXA": "XLY", "FOX": "XLY", "VIAV": "XLK",

    # Consumer Staples delistings
    "KFT": "XLP", "SLE": "XLP", "HNZ": "XLP", "WAG": "XLP", "DLM": "XLP",
    "RAI": "XLP", "WFM": "XLP", "SVU": "XLP", "DF": "XLP", "MJN": "XLP",
    "AVP": "XLP",

    # Industrials / transport delistings
    "UTX": "XLI", "RTN": "XLI", "COL": "XLI", "LLL": "XLI", "PCP": "XLI",
    "GY": "XLI", "TYC": "XLI", "TKR": "XLI", "DLPH": "XLY", "TEN": "XLY",
    "TRW": "XLY", "WHR": "XLY", "BNI": "IYT",
    "AMR": "IYT", "UAUA": "IYT", "CAL": "IYT", "NWAC": "IYT", "LCC": "IYT",
    "NAV": "XLI",

    # Materials / metals / REIT / telecom misc
    "GGP": "IYR",

    # Modern delisted / acquired that appear in MODERN_TICKERS
    "ATVI": "XLY", "SGEN": "IBB", "HZNP": "IBB", "PXD": "XLE", "ALXN": "IBB",
    "CELG": "IBB", "WBD": "XLY",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIC_TO_ETF: broad industry-code → sector-ETF mapping used by
# CRSPSource.fetch_sector_mapping. Two-digit SIC prefix is the primary key;
# SIC3_TO_ETF_OVERRIDES refines specific 3-digit ranges (e.g. 283 pharma
# overrides the 28 chemicals bucket).
#
# Reference: SEC SIC code list (https://www.sec.gov/info/edgar/siccodes.htm)
# and CRSP header info documentation for hsiccd.
# ─────────────────────────────────────────────────────────────────────────────
SIC_TO_ETF: Dict[int, str] = {
    # Agriculture, mining, construction
    10: "XLB", 12: "XLE", 13: "XLE", 14: "XLB",
    15: "XHB", 16: "XLI", 17: "XLI",
    # Manufacturing 20-39
    20: "XLP", 21: "XLP", 22: "XLY", 23: "XLY", 24: "XLB",
    25: "XLY", 26: "XLB", 27: "XLY", 28: "XLB", 29: "XLE",
    30: "XLB", 31: "XLY", 32: "XLB", 33: "XLB", 34: "XLI",
    35: "XLI", 36: "XLK", 37: "XLI", 38: "XLV", 39: "XLY",
    # Transportation, communications, utilities 40-49
    40: "IYT", 41: "IYT", 42: "IYT", 44: "IYT", 45: "IYT",
    46: "XLI", 47: "IYT", 48: "IYZ", 49: "XLU",
    # Trade 50-59
    50: "XLY", 51: "XLP", 52: "XHB", 53: "XLY", 54: "XLP",
    55: "XLY", 56: "XLY", 57: "XLY", 58: "XLY", 59: "XLY",
    # Finance/insurance/RE 60-67
    60: "XLF", 61: "XLF", 62: "XLF", 63: "XLF", 64: "XLF",
    65: "IYR", 67: "XLF",
    # Services 70-89
    70: "XLY", 72: "XLY", 73: "XLK", 75: "XLY", 78: "XLY",
    79: "XLY", 80: "XLV", 82: "XLY", 87: "XLI",
}

# 3-digit refinements. Keyed by SIC // 10.
SIC3_TO_ETF_OVERRIDES: Dict[int, str] = {
    # Pharmaceuticals & biotech (overrides 28 chemicals)
    283: "XLV",
    # Computer hardware (overrides 35 industrial machinery)
    357: "XLK",
    # Motor vehicles (overrides 37 transport equipment)
    371: "XLY",
    # Aircraft / aerospace / defense (keeps 37 → XLI)
    372: "XLI", 376: "XLI",
    # Medical & optical instruments (keeps 38 → XLV)
    384: "XLV", 386: "XLK",
    # Telephone & communication services
    481: "IYZ", 482: "IYZ", 483: "IYZ", 484: "IYZ", 489: "IYZ",
    # Gas, electric, water utilities (keeps 49 → XLU)
    491: "XLU", 492: "XLU", 493: "XLU", 494: "XLU",
    # Banks
    602: "XLF", 603: "XLF", 606: "XLF",
    # Life & health insurance
    631: "XLF", 632: "XLF", 633: "XLF",
    # Real estate investment trusts & operators
    651: "IYR", 679: "IYR",
    # Computer services (keeps 73 → XLK, explicit for clarity)
    737: "XLK",
    # Restaurants (keeps 58 → XLY)
    581: "XLY",
    # Health services (keeps 80 → XLV)
    806: "XLV", 807: "XLV", 809: "XLV",
}

DATA_SOURCES = ["yfinance", "crsp"]


@dataclass
class FactorConfig:
    model_type: str = "pca"
    pca_lookback: int = 252
    pca_n_components: Optional[int] = 15
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252


@dataclass
class OUConfig:
    estimation_window: int = 60
    kappa_min: float = 8.4
    mean_center: bool = True


@dataclass
class SignalConfig:
    s_bo: float = 1.25
    s_so: float = 1.25
    s_sc: float = 0.50
    s_bc: float = 0.75
    s_limit: float = 4.0


@dataclass
class VolumeConfig:
    enabled: bool = False
    trailing_window: int = 10


@dataclass
class PairsConfig:
    pvalue_threshold: float = 0.10
    max_pairs: int = 20
    min_half_life: float = 1.0
    max_half_life: float = 126.0
    lookback_window: int = 252
    auto_select: bool = True


@dataclass
class BacktestConfig:
    initial_equity: float = 1_000_000.0
    leverage_long: float = 2.0
    leverage_short: float = 2.0
    tc_bps: float = 5.0
    hedge_instrument: str = "SPY"
    risk_free_rate: float = 0.02
    dt: float = 1.0 / 252.0


@dataclass
class VolTargetConfig:
    """Volatility-targeted position sizing (Extension)."""
    enabled: bool = False
    # Clamp the scale factor (target_sigma / sigma_eq) to this range so that
    # no single position blows up or disappears relative to equal-notional.
    floor_multiplier: float = 0.2
    cap_multiplier: float = 5.0


@dataclass
class HMMConfig:
    """HMM regime detection to gate entries (Extension)."""
    enabled: bool = False
    n_states: int = 2
    # Days of history used to fit the HMM before any trading begins.
    training_window: int = 252
    # Rolling window for computing cross-sectional vol features.
    feature_window: int = 20
    # Minimum P(favorable regime | data up to t) required to open new trades
    # (only used in binary gating mode).
    entry_threshold: float = 0.5
    # If True, label the HIGH cross-sectional residual-vol state as favorable
    # (mean-reversion typically harvests dispersion). If False, label the LOW
    # vol state as favorable (calm = mean-reverting hypothesis).
    favorable_high_vol: bool = True
    # If True, scale per-position notional by p_fav instead of binary gating.
    # `entry_threshold` is ignored when soft_gate is True; the floor below
    # caps how aggressively the book is de-risked.
    soft_gate: bool = True
    soft_gate_floor: float = 0.2


@dataclass
class Config:
    factor: FactorConfig = field(default_factory=FactorConfig)
    ou: OUConfig = field(default_factory=OUConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    vol_target: VolTargetConfig = field(default_factory=VolTargetConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    trading_mode: str = "statarb"
    data_source: str = "yfinance"
    start_date: str = "1997-01-01"
    end_date: str = "2007-12-31"
    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
