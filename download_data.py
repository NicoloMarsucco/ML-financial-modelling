
import wrds
import pandas as pd

###################
# Connect to WRDS #
###################

db = wrds.Connection()

###################
# CRSP: returns and prices of stocks and factor to adjust forecasts (CFACSHR)
# - Common stocks (share code 10 and 11) in stock exchanges of NYSE, AMEX, and NASDAQ (exchange code 1, 2, and 3)
# - If closing price is not available on any given trading day, the number in the price field has a negative sign to indicate that 
#it is a bid/ask average and not an actual closing price. Please note that in this field the negative sign is a symbol and that the value
# of the bid/ask average is not negative.
###################
if 1==0:
     crsp = db.raw_sql("""
                         select a.permno, a.cusip, a.date, a.cfacshr, abs(a.prc) as price, b.shrcd, b.exchcd, a.ret              
                         from crsp.dsf as a
                         left join crsp.msenames as b
                         on a.permno=b.permno
                         and b.namedt<=a.date
                         and a.date<=b.nameendt
                         where  a.cusip != ''
                         and a.date>='1985-01-01'
                         and (b.exchcd = '1' or b.exchcd = '2'or b.exchcd ='3')
                         and (b.shrcd = '10' or b.shrcd =  '11')
                    """)

     crsp.to_csv('data/crsp.csv')

###################
#IBES: analysts average estimates and actual values
###################
ibes_summary = db.raw_sql("""
                    select ticker, cusip, cname, fpedats, statpers, meanest, fpi, numest, actual, anndats_act
                    from ibes.statsum_epsus
                    where  cusip != ''
                    and usfirm='1'
                    and fpedats>='1985-01-01'
                    and (fpi='1' or fpi='2' or fpi='6' or fpi='7' or fpi='8')
                    """)

ibes_summary.to_csv('data/ibes_summary.csv')
if 1==0:
     ibes_actual = db.raw_sql("""
                         select ticker, pends, value, anndats
                         from tr_ibes.actu_epsus 
                         where  cusip != ''
                         and usfirm='1'
                         and pdicity='QTR'
                         and pends>='1985-01-01'
                         """)

     ibes_actual.to_csv('data/ibes_actual.csv')

     ###################
     #Financial Ratio
     ###################

     finratio = db.raw_sql("""
                         select *
                         from  wrdsapps_finratio_ibes.firm_ratio_ibes
                         where  cusip != ''
                         and public_date>='1985-01-01'
                         """)

     finratio.to_csv('data/finratio.csv')


     ###################
     # Download from Federal Reserve Bank of Philadelphia #
     ###################

     ###################
     #Real GDP
     ###################

     url = (
          "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/routputmvqd.xlsx?la=en&hash=403C8B9FD72B33F83C1EE5C59D015C86"
          )
     df = pd.read_excel(url)
     df.to_csv('data/real_GDP_FED.csv')

     ###################
     #IPT
     ###################

     url = (
          "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/iptmvmd.xlsx?la=en&hash=E53F4C735866E2366E50511D5C9CCADE"
          )
     df = pd.read_excel(url)
     df.to_csv('data/IPT_FED.csv')

     ###################
     # Real Personal Consumption
     ###################

     url = (
          "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/rconmvqd.xlsx?la=en&hash=9F7B44DB227E6A620629495229CD93BB"
          )
     df = pd.read_excel(url)
     df.to_csv('data/real_personal_consumption_FED.csv')


     ###################
     # Unemployement rate
     ###################

     url = (
          "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/rucqvmd.xlsx?la=en&hash=FF1D4C67E144D916C1986A8EEDC4B42A"
          )
     df = pd.read_excel(url)
     df.to_csv('data/Unemployment_FED.csv')
