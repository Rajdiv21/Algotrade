import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file


# ==================== (Set working directory)
# print("current working directory", os.getcwd())
#
# # =================== (Import dataset)
# Infy = pd.read_csv('C:/Users/hhhg/PycharmProjects/pythonProject4/INFY1.csv')
# print(Infy)
# print(Infy.head())
# print(Infy.tail())
#
# # ====================== (Filter the required data)
# infy_close = Infy[['Date', 'Close']]
# infy_close.set_index('Date', inplace=True)
# print(infy_close)
#
# # ======================= (Check the plot)
# plt.plot(infy_close)
# plt.show()
#
# # ====================== (Transform the plot)
# plt.figure(figsize=(20, 5))
# plt.plot(infy_close, 'b')
# plt.plot(infy_close, 'ro')
# plt.grid(True)
# plt.title('Infosys Closing Price Chart')
# plt.xlabel('Trading Days')
# plt.ylabel('Infosys Closing Price')
# plt.show()
#
# # ===================== (Getting Data)
# infy2 = pd.read_csv('C:/Users/hhhg/PycharmProjects/pythonProject4/INFY1.csv')
# print(infy2)
# infy2 = infy2[['Date', 'Open', 'Close']]
# infy2.set_index('Date', inplace=True)
# print(infy2)
#
# # ====================== (Plotting Data)
# plt.figure(figsize=(14, 5))
# plt.plot(infy2["Close"], lw=1.5, label='Close')
# plt.plot(infy2["Open"], lw=1.5, label='Open')
# plt.plot(infy2, 'ro')
# plt.grid(True)
# plt.legend(loc=0)
#
# # ====================== (Tighten the figure margins)
# plt.axis('tight')
# plt.xlabel('Time')
# plt.ylabel('Index')
# plt.title('Open-Close Plot')
# plt.show()
#
# # ================== (Scatter Plot)
# y = np.random.standard_normal((100, 2))
# plt.figure(figsize=(7, 5))
# plt.scatter(y[:, 0], y[:, 1], marker='o')
# plt.grid(True)
# plt.xlabel('1st dataset')
# plt.ylabel('2nd dataset')
# plt.title('Scatter Plot')
# plt.show()
#
# # =================== (Histogram)
# np.random.seed(100)
# y = np.random.standard_normal(size=1000)
# plt.figure(figsize=(10, 5))
# plt.hist(y, label=['Return Distribution'])
# plt.grid(True)
# plt.legend(loc=0)
# plt.ylabel('Frequency')
# plt.xlabel('Returns in Percentage')
# plt.title('Histogram')
# plt.show()
#
# # =========== (3D Plotting)
# strike_price = np.linspace(50, 150, 25)
# time= np.linspace(0.5, 2, 25)
#
# strike_price, time = np.meshgrid(strike_price, time)
# print(strike_price, time[:])
#
# # ============ (Generate Implied Volatility)
# implied_volatility = (strike_price - 100) ** 2 / (100 * strike_price)/time
#
# # ============ (Required package for 3D plot)
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(9, 6))
#
# # ============== (enabling 3D axes)
# axis = fig.gca(projection='3d')
#
# # =============== (To plot the surface and passing the required arguments)
# surface = axis.plot_surface(strike_price, time, implied_volatility, rstride=1,
#                             cstride=1, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=False)
#
# axis.set_xlabel('stike price')
# axis.set_ylabel('time-to-maturity')
# axis.set_zlabel('implied volatility')
#
# fig.colorbar(surface, shrink=0.5, aspect=5)
# plt.show()
#
# # =========== (Plotting Candlesticks)
#
# df = pd.read_csv('C:/Users/hhhg/PycharmProjects/pythonProject4/INFY1.csv', index_col=0)
# print(df)
# print(df.head())
# print(df.tail())
#
# # ========== (Importing Library)
# # from bokeh.plotting import figure, show, output_file
#
# # ========== (Indexing)
# w = 12*60*60*1000
# df.index = pd.to_datetime(df.index)
# #
# # # ========== (Candlesticks patterns in Code)
# adv = df.close > df.open
# dec = df.open > df.close
# #
# # # ========= (Interactions needed in candlestick graph)
# TOOLS = "pan, wheel_zoom, box_zoom, reset, save"
# #
# # # Pan: It helps you pan/move the plot
# # # Wheel Zoom: You can zoom in using the wheel of your mouse
# # # Box Zoom: You can zoom in by creating a box on the specific area of the plot.
# # # Reset: If you want to reset the visualisation of the plot
# # # Save: Saving the plot (entire or the part which you want) as an image file
# #
# # # ========= (Passing arguments for bokeh plot)
# p = figure(x_axis_type="datetime", tools=TOOLS,
#             plot_width=1000, tittle="INFY Candlestick")
# #
# # # ========== (Importing Library)
# from math import pi
# p.xaxis.major_label_orientation = pi/4
# #
# # # The orientation of major tick labels can be controlled with the major_label_orientation property.
# # # This property accepts the values "horizontal" or "vertical" or a floating point number that gives
# # # the angle (in radians) to rotate from the horizontal.
# #
# # # ========== (Grid lines in the plot)
# p.grid.grid_line_alpha = 0.3
#
# # ========== (Configure and add segments to the plot)
# p.segment(df.index, df.High, df.index, df.Low, color="red")
#
# # ========== (Adding vbar to the plot)
# p.vbar(df.index[adv],w, df.open[adv],
#        fill_color="#1ED837", line_color="black")
#
# p.vbar(df.index[dec],w, df.open[dec],
#        fill_color="#F2583E", line_color="black")
#
# # ========== (Simple HTML document for bokeh visualisation)
# output_file("candlestick.html", tittle = "Py.candlestick")
#
# show(p)
#
# # =========== (Calculating Moving Average and Standard Deviation)
# def Bollinger_bands (data, n):
#
#     MA = data['Close'].rolling(window=n).mean()
#     SD = data['Close'].rolling(window=n).std()
#
#     data['Lower_BB'] = MA - (2 * SD)
#     data['Upper_BB'] = MA + (2 * SD)
#
#     return data
#
# # =========== (Importing Nifty Data)
# nifty = pd.read_csv('C:/Users/hhhg/PycharmProjects/pythonProject4/NIFTY 1.csv')
# print(nifty.head())
#
# # =========== (Calling Bollinger Bands function)
# n = 11
# nifty_bb = Bollinger_bands(nifty, n)
# print(nifty_bb)
# print(nifty_bb.tail())
#
# # =========== (Plotting Bollinger Bands for Nifty)
# plt.figure(figsize=(10, 5))
# plt.plot(nifty_bb.Close)
# plt.plot(nifty_bb.Lower_BB)
# plt.plot(nifty_bb.Upper_BB)
# plt.grid(True)
#
# plt.show()
#
# # =========== (Importing Nifty Data)
# nifty = pd.read_csv('C:/Users/hhhg/PycharmProjects/pythonProject4/INFY1.csv')
# print(nifty.head())
#
# # =========== (Calling Bollinger Bands function)
# n = 3
# nifty_bb = Bollinger_bands(nifty, n)
# print(nifty_bb)
# print(nifty_bb.tail())
#
# # =========== (Plotting Bollinger Bands for Nifty)
# plt.figure(figsize=(10, 5))
# plt.plot(nifty_bb.Close)
# plt.plot(nifty_bb.Lower_BB)
# plt.plot(nifty_bb.Upper_BB)
# plt.grid(True)
#
# plt.show()
#
# # ===============  (Use of Lambda function)
# def product(x, y):
#     return x * y
#
# print(product(26, 75))
# print(product(83, 29))
#
# def my_operation(x, y, z):
#     return x + y - z
#
# print(my_operation(25, 131 ,81))
# print(my_operation(112, 14, 79))
#
# # =============== (Use of map function in lambda)
# list_1 = [11, 21, 31, 41]
# list_2 = [111, 211, 311, 411]
# list_3 = [1111, 2111, 3111, 4111]
#
# a = list(map(lambda x, y: x + y, list_2, list_3))
# print(a)
#
# b = list(map(lambda x, y, z: x + y + z, list_1, list_2, list_3))
# print(b)
#
# c = list(map(lambda y, z: y + z, list_2, list_3))
# print(c)
#
# # ============== (Use of filter function in lambda)
# fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
# d = list(filter(lambda x: x > 5, fib))
# print(d)
#
# # ================ (Understanding numpy and arrays)
# stock_values = [50.1, 50.3, 51.2, 53.1, 54.2,
#                 52.3, 55.1, 56.5, 58.5, 59.1]
#
# # ================ (Converting list into an array)
# SV = np.array(stock_values)
# print(SV)
# print(type(SV))
#
# # ================ (Converting tuple into an array)
# stockvalues = (50.1, 50.3, 51.2, 53.1, 54.2,
#                 52.3, 55.1, 56.5, 58.5, 59.1)
#
# S_V = np.array(stockvalues)
# print(S_V)
# print(type(S_V))
#
# # =============== (Understanding arange)
# a = np.arange(0, 15, 3)
# print(a)
#
# b = np.arange(1.3, 28.4, 2, float)
# print(b)
#
# c = np.arange(0.1, 21.5, 1, int)
# print(c)
#
# # ============= (Understanding linspace)
# d = np.linspace(11, 20, endpoint=False)
# print(d)
#
# e = np.linspace(5, 25, endpoint=True, retstep=False)
# print(e)
#
# f = np.linspace(1, 10, 8, endpoint=True, retstep=True)
# print(f)
#
# # ============ (Dimensionality of arrays, 0 dimension)
# a = np.array(100)
# print("a:", a)
# print("The dimension of array 'a' is", np.ndim(a))
# print("The datatype of array 'a' is", a.dtype)
#
# scaler_array = np.array("one element")
# print(scaler_array, np.ndim(scaler_array), scaler_array.dtype)
#
# # ============ (One dimensional arrays)
# one_d_array = np.array(["one_element", "second_element"])
# print(one_d_array, np.ndim(one_d_array), one_d_array.dtype)
#
# m = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
# n = np.array([2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8])
# print("m:", m)
# print("n:", n)
# print("Dimension of 'm': ", np.ndim(m))
# print("Dimension of 'n': ", np.ndim(n))
# print("Type of 'm': ", m.dtype)
# print("Type of 'n': ", n.dtype)
#
# # ============= (Two dimensional arrays)
# twod_array = np.array([["row1col1", "row1col2", "row1col3"],
#                ["row2col1", "row2col2", "row2col3"]])
# print(twod_array)
# print("Dimension of 'twod_array': ", np.ndim(twod_array))
#
# studentdata = np.array([["Name", "Year", "Marks"],
#                         ["Div", "1995", "97.5"],
#                         ["Sat", "1996", "92"],
#                         ["Suk", "1994", "98"]])
#
# print(studentdata)
# print("Dimension of 'Studentsdata': ", np.ndim(studentdata))
#
# studentdata1 = {
#     "Name": ["Div", "Sat", "Suk"],
#     "Year": ["1995", "1996", "1994"],
#     "Marks": ["97.5", "91", "98"]
# }
# studentdata1_df = pd.DataFrame(studentdata1)
# print(studentdata1_df)
# print(np.mean(studentdata1_df.Marks))
#
# # =============== (Three dimensional arrays)
# d = np.array([[[121, 125], [324, 331]],
#               [[151, 157], [391, 393]],
#               [[168, 171], [345, 397]]])
#
# print(d)
# print("Dimension of 'd':", np.ndim(d))
#
# # =============== (Shape of an array)
# t = np.array([[11, 22, 33],
#               [12, 24, 36],
#               [13, 26, 39],
#               [14, 28, 42],
#               [15, 30, 45],
#               [16, 32, 48],
#               [17, 34, 51],
#               [18, 36, 54],
#               [19, 38, 57],
#               [20, 40, 60]])
#
# print(t)
# print(t.shape)
# t.shape = (15, 2)
# print(t)
#
# # ============== (Indexing in one-dimensional array)
# A = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
# print(A[0])
# print(A[-3])
# print(A[6])
#
# # ============== (Indexing in two-dimensional array)
# B = np.array([[1, 3, 5],
#               [7, 9, 11],
#               [13, 15, 17]])
# print(B)
# print(B[0, 2])
# print(B[1, 2])
# print(B[2, 0])
#
# # ============== (Slicing an one-dimensional array)
# C = np.array([1, 3, 5, 7, 11, 13, 17, 19, 23])
# print(C[:])
# print(C[3:])
# print(C[:7])
# print(C[7:23])
#
# # ============= (Slicing a two-dimensional array)
# H = np.array([
#     [1, 2, 3, 4],
#     [10, 20, 30, 40],
#     [11, 21, 31, 41],
#     [20, 30, 40, 50],
#     [21, 31, 41, 51]
# ])
# print(H)
# print(H[1, ])
# print(H[:, 1])
# print(H[:2, ])
# print(H[:, 1:3])
# print(H[:, 3:])
#
# # ============== (Using step for Indexing/Slicing)
# J = np.arange(35).reshape(5, 7)
# print(J)
# print(J[::2, ])
# print(J[:, 1:5:3])
#
# # ============== (Array of ones and zeros)
# O = np.ones((4,4))
# print(O)
# O = np.ones((4,4), dtype=int)
# print(O)
# O = np.ones((3,3), dtype=float)
# print(O)
#
# # ============= (Identity function)
# I = np.identity(3, dtype=int)
# print(I)
# L = np.identity(4)
# print(L)
#
# # ============= (Array operations with scalar)
# my_list = [1, 2, 3.5, 4.6, 5.658, 6.755, 7.81]
# V = np.array(my_list)
# print(V)
# V_a = V + 1.5
# print(V_a)
# V_b = V_a - 2.3
# print(V_b)
# V_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# vc = V_2 * 10
# print(vc)
#
# # ============= (2D array operation with another 2D array)
# A = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
# B = np.ones((3, 3))
# C = np.ones((4, 4))
# print(A)
# print(B)
# print(C)
# print(A + B)
# print(A * B)
# print(A - B)
#
# A1 = np.array([[1, 2, 3], [4, 5, 6]])
# A2 = np.array([[1, 0, -1], [0, 1, -1]])
# print("Multiplying of two arrays", A1 * A2)
#
# # ============== (Vectorization and Broadcasting in arrays)
# b1 = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
# b2 = np.array([1, 2, 3])
# print("Multiplication with broadcasting:")
# print(b1 * b2)
# print("Addition with broadcasting")
# print(b1 + b2)
#
# c = np.array([[1, 2, 3], ] * 3)
# print(c)
#
# d = np.array([1, 2, 3])
# print(d[:, np.newaxis])
# print(c * d[:, np.newaxis])
#
# # ================ (Logical Operators)
# a = np.array([[True, True], [False, False]])
# b = np.array([[True, False], [True, False]])
# print(np.logical_or(a, b))
# print(np.logical_and(a, b))

# ================= (Understanding Series)
My_series = pd.Series([10, 20, 30, 40, 50, 60])
print(My_series, My_series.dtype)
print(My_series)

my_series = pd.Series([10.1, 20.2, 30.3, 40.4, 50.5])
print(my_series)

mixed_series = pd.Series([10, 20.3, 'div', 31.5])
print(mixed_series)

countries = ['India', 'US', 'China', 'Russia', 'Japan']
leaders = ['Modi', 'Trump', 'Xi', 'Putin', 'Abe']
S = pd.Series(leaders, index=countries)
print(S)

stock_set = ['Infosys', 'RIL', 'Sterling_Wilson', 'Coromandel_Int']
S1 = pd.Series([1400, 2000, 285, 735], index=stock_set)
print(S1)
print("\n")

stock_set2 = ['Infosys', 'RIL', 'Sterling_Wilson', 'Coromandel_Int']
S2 = pd.Series([1450, 2021, 300, 830], index=stock_set2)
print(S2)
print("\n")

print(S1 + S2)

# =============== (Understanding Functions)
mys = pd.Series([10, 20, 30, 40, 50, 60])
print(mys.index)

stock_set3 = ['Infosys', 'RIL', 'IEX', 'Coromandel_Int']
S3 = pd.Series([1420, 2100, 330, 750], index=stock_set3)
print(S3)
print(S2 + S3)
print((S2 + S3).isnull())
print((S2 + S3).dropna())
print((S2 + S3).fillna(1))

# ============== (pandas.series.apply)
msp = pd.Series([10, 20, 30, 40, 50])
print(msp)
print(msp.apply(np.sin))
print(msp.apply(np.tan))

# ============= (Creating Dataframe)
my_portfolio = {
    "Stock_name" : ["Infosys", "RIL", "IEX", "Sterling_Wilson", "Coromandel_Int"],
    "Quantity_owned" : [1050, 500, 2500, 3200, 1000],
    "Average_price" : ["1380", "1950", "310", "245", "725"]
}
my_portfolio_frame = pd.DataFrame(my_portfolio)
print(my_portfolio_frame)

# ============ (Customize index of the dataframe)
ordinals = ["first", "second", "third", "fourth", "fifth"]
my_portfolio_frame = pd.DataFrame(my_portfolio, index=ordinals)
print(my_portfolio_frame)

my_portfolio_frame = pd.DataFrame(my_portfolio,
                                  columns=["Quantity_owned",
                                  "Average_price"],
                                  index=my_portfolio["Stock_name"])
print(my_portfolio_frame)

# ============ (Access a column in dataframe)
print(my_portfolio_frame["Quantity_owned"])
print(my_portfolio_frame["Average_price"])

# =========== (Extracting financial markets data)
print("current working directory", os.getcwd())
iex = pd.read_csv('E:/PycharmProjects/pythonProject4/IEX1.csv')
print(iex.head())
print(iex.tail())

iex_new = iex.drop(['Volume'], axis=1)
print(iex_new)
print(iex_new.drop(iex_new.index[[3, 4]]).head())
print(iex_new.drop(iex_new.index[[62, 61]]).tail())

iex_new = iex_new.sort_values(by='High', ascending=False)
print(iex_new.head())












