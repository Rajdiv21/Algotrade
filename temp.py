#IMPORTING LIBRARIES
import requests
import json
from matplotlib import pyplot as plt
from datetime import datetime

# IGNORE THIS PART ===========
url = 'http://alphabet.researchandranking.com/stockOverview'
body = {'stock_symbol': '3MINDIA'}
headers = {'content-type': 'application/json',
           'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImFwcHN1cHBvcnRAcmVzZWFyY2hhbmRyYW5raW5nLmNvbSIsInRpbWVzdGFtcCI6IjIwMjEtMDMtMTggMTI6NDk6MjMuNzYyNTA0KzA1OjMwIn0.XB6qJtHBLyBRw2IBt4KksWVgFKLS4SibVi7DwhkTQRU'}

myRequest = requests.post(url, data=json.dumps(body), headers=headers)
historicalData = myRequest.json()["candlestickData"]

dateList = []
openList = []
closeList = []

for item in historicalData:
    dateObj = datetime.strptime(item[0], "%Y-%m-%d")
    dateList.append(dateObj.strftime("%d %b"))
    openList.append(item[1])
    closeList.append(item[2])

# ================================================

plt.plot(dateList[-30:], openList[-30:], 'g', label='Open Price', linewidth=1)
plt.plot(dateList[-30:], closeList[-30:], 'c', label='Closing Price', linewidth=2)
plt.title('Historical Data')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.xticks(rotation=90, fontsize=6)
plt.show()
