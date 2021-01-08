from flask import Flask
from flask import render_template
from flask import request
from flask_bootstrap import Bootstrap


app=Flask(__name__)
Bootstrap(app)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

url = "creditcard.csv"

dataset = pd.read_csv(url)
creditcard = dataset
cc = dataset.shape

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 30].values

#describe function
Od=dataset.describe()

#histogram
#creditcard.hist(figsize = (20,20))
#Od1=plt.show()

#2d Scatter Plot
#sns.set_style("whitegrid")
#sns.FacetGrid(creditcard, hue="Class", height = 6).map(plt.scatter, "Time", "Amount").add_legend()
#Od2=plt.show()

#sns.set_style("whitegrid")
#sns.FacetGrid(creditcard, hue="Class", height = 6).map(plt.scatter, "Amount", "Time").add_legend()
#Od3=plt.show()

#3d Scatter Plot
#FilteredData = creditcard[['Time','Amount', 'Class']]
#plt.close();
#sns.set_style("whitegrid");
#sns.pairplot(FilteredData, hue="Class", height=5);
#Od4=plt.show()

#1D Scatter Plot
#creditCard_genuine = FilteredData.loc[FilteredData["Class"] == 0]
#creditCard_fraud = FilteredData.loc[FilteredData["Class"] == 1]
#plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o")
#plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o")

#Od5=plt.show()
#X-axis: Amount


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 

n_errors = (y_pred != y_test).sum() 


# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
acc = accuracy_score(y_test, y_pred)
acc1=acc*100
acc1=round(acc1,3)


# Precision or positive predictive value
#PPV = TP/(TP+FP)
prec = precision_score(y_test, y_pred)
prec = prec*100
prec =round(prec,3) 


# Sensitivity, hit rate, recall, or true positive rate
#rec = TP/(TP+FN)
rec = recall_score(y_test, y_pred)
rec =rec*100
rec =round(rec,3) 
 
  
f1 = f1_score(y_test, y_pred) 
 
  
MCC = matthews_corrcoef(y_test, y_pred)
MCC =MCC*100
MCC =round(MCC,3) 





@app.route('/')
def index():
	return render_template('index.html', data={'output1': cc})

@app.route('/trans1', methods=['POST'])
def trans1():
	username="Not Found"
	arr1 = np.array([[-1.12344134, -1.11126229, -3.539774  , -0.79035282,  1.12189503,
	-1.60332892,  0.99071823,  1.27381779, -0.03167408,  0.51166851,
	-0.95128711,  1.19957763,  1.12183845, -0.47263665,  0.35269883,
	-0.51731272, -0.57906528,  0.7316591 , -0.70930429, -0.7174261 ,
	4.24420975,  1.48656871, -0.5616327 , -2.4740302 , -0.16753636,
	-0.65132818,  2.14030774, -0.86083354,  0.85004075,  6.45615328]])
	y_pred = classifier.predict(arr1)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username,'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})	

@app.route('/trans2', methods=['POST'])
def trans2():
	username="Not Found"
	arr2 = np.array([[-0.81567905, -0.7680087 ,  0.6303559 ,  1.53278435,  0.55306741,
       -0.24857578,  0.71313533,  0.17998226,  0.18746037,  1.19399836,
        0.60132845,  0.6301896 ,  0.8592295 , -1.32455058, -1.054175  ,
       -2.67113516, -1.49304517,  0.57951045, -0.87877879,  0.67651426,
        0.15011029, -0.52911992, -0.52852707, -0.16812967,  0.32385103,
       -0.35165871, -1.44641771, -1.59045133, -1.45579856, -0.32071674]])
	y_pred = classifier.predict(arr2)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"

	return render_template('next.html', data={'des':"Transaction" ,'output1': username,'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})





@app.route('/trans4', methods=['POST'])
def trans4():
	username="Not Found"
	arr4 = np.array([[ 0.56686083, -0.88170219,  0.2377539 , -0.3867855 , -0.92783987,
	1.52020169, -0.54195445,  0.42495841,  0.01426442,  0.45659251,
	0.48100726, -0.40186998,  0.26179774, -0.31815765,  0.29159924,
	-0.81617787, -0.00250781, -1.18935156,  0.30738781,  1.00185729,
	0.14554203, -0.44246233, -0.44002889, -0.78596435, -2.34973733,
	0.31417595, -0.0515969 ,  1.78748335,  1.69201951, -0.34872891]])
	y_pred = classifier.predict(arr4)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans5', methods=['POST'])
def trans5():
	username="Not Found"
	arr5 = np.array([[ 1.12983857e+00,  1.11743233e-01,  1.64318663e+00, -3.36845522e+00,
	4.45703153e+00, -6.14464525e-01, -6.61861600e-01, -2.34386811e+00,
	7.86488779e-01, -3.30121957e+00, -1.71890259e+00,  1.17917273e+00,
	-6.17632001e+00, -9.25725131e-01, -4.10934124e+00, -6.78830804e-02,
	-5.97933385e+00, -9.58907902e+00, -3.86983952e+00,  2.37745714e+00,
	4.94255712e-01,  1.47577496e+00,  1.42916707e+00,  9.97207712e-02,
	8.79310136e-01, -2.86345096e-01,  1.32651484e+00,  8.71139049e-01,
	-5.53512234e-03, -3.49686592e-01]])
	y_pred = classifier.predict(arr5)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans6', methods=['POST'])
def trans6():
	username="Not Found"
	arr6 = np.array([[ 0.12983857e+00,  3.11743233e-01,  1.64318663e+00, -3.36845522e+00,
	4.45703153e+00, 6.14464525e-01, 6.61861600e-01, 2.34386811e+00,
	7.86488779e-01, -3.30121957e+00, -1.71890259e+00,  1.17917273e+00,
-	6.17632001e+00, -9.25725131e-01, 4.10934124e+00, 6.78830804e-02,
	5.97933385e+00, -9.58907902e+00, 3.86983952e+00,  -2.37745714e+00,
	-4.94255712e-01,  -1.47577496e+00,  -1.42916707e+00,  9.97207712e-02,
	8.79310136e-01, 2.86345096e-01,  1.32651484e+00,  8.71139049e-01,
	5.53512234e-03, 3.49686592e-01]])
	y_pred = classifier.predict(arr6)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
	return render_template('next.html', data={'des':"Transaction" ,'output1': username,'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})


@app.route('/trans7', methods=['POST'])
def trans7():
	username="Not Found"
	arr7 = np.array( [[ 0.64097398 , 1.07050715 ,-1.196556 ,  -1.40497357 ,-1.77841841 , 0.61712233 ,
	2.73508027 , -1.43280806 ,  0.72850108 , -1.10196198 , 1.40325102 , -0.23593605 ,
	-0.70022316 ,  0.4010508 ,  -0.35926203 , 0.38047459 , -0.57924879 ,  0.41736192 ,
	-0.12829918 ,-0.45845755 , -0.23103513 , -0.12541209 ,-0.05815003 , 0.35539377 ,
	1.13683559 ,-0.52834671 , -0.3416334 ,  0.07452915 , -0.10844711 , 0.05429518 ]])
	y_pred = classifier.predict(arr7)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans3', methods=['POST'])
def trans3():
	username="Not Found"
	arr3 = np.array( [[ -1.12853806,  -5.2475586 ,   3.8129177 ,  -8.74613397,
         6.3034609 ,  -7.2207322 ,  -2.12469427, -10.25860302,
         5.61612012,  -6.44194792, -11.75290342,   6.64685567,
       -13.07212716,   1.18580093, -14.27818239,   1.03931203,
       -12.497142  , -24.20955003,  -8.96924358,   3.5282345 ,
        -0.32061641,   3.37585659,   0.50571697,   0.06852098,
         0.78976614,   0.30227764,   0.6843599 ,   0.40502916,
        -1.47344259,   0.11933768]])
	y_pred = classifier.predict(arr3)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans8', methods=['POST'])
def trans8():
	username="Not Found"
	arr8 = np.array( [[-1.12546318, -0.17702889,  0.2814068 ,  0.72843145, -0.19000884,
        0.56024895,  0.99585674,  0.09901454,  0.57385727, -0.38149677,
       -0.32420402,  1.86571772,  0.27020573, -1.59885759,  0.8624645 ,
        1.35859687, -0.92622831,  0.68955278, -1.89741099, -1.54970311,
       -0.39150505,  0.0922878 ,  0.29347721,  0.50419556, -1.77525749,
       -2.94743484,  0.03294947,  0.57649973,  0.56212615, -0.34481838]])
	y_pred = classifier.predict(arr8)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
   		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans9', methods=['POST'])
def trans9():
	username="Not Found"
	arr9 = np.array( [[-1.30862944,  0.14703228,  1.04601163, -1.08857033,  2.69347061,
       -0.79004481, -0.73860054, -1.77879032,  0.46488012, -1.85059375,
       -2.50899017,  2.31987567, -3.65896053, -0.17008884, -4.94601156,
        0.83584432, -2.82890265, -5.79984868, -3.03935216, -1.10873228,
        0.34349689,  0.35754137, -0.87239717,  0.1485745 ,  0.3096504 ,
        0.7068273 , -0.27442868,  1.42877853,  0.94020218, -0.35271925]])
	y_pred = classifier.predict(arr9)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
    		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/trans10', methods=['POST'])
def trans10():
	username="Not Found"
	arr10 = np.array( [[-1.12546318, -0.17702889,  0.2814068 ,  0.72843145, -0.19000884,
        0.56024895,  0.99585674,  0.09901454,  0.57385727, -0.38149677,
       -0.32420402,  1.86571772,  0.27020573, -1.59885759,  0.8624645 ,
        1.35859687, -0.92622831,  0.68955278, -1.89741099, -1.54970311,
       -0.39150505,  0.0922878 ,  0.29347721,  0.50419556, -1.77525749,
       -2.94743484,  0.03294947,  0.57649973,  0.56212615, -0.34481838]])
	y_pred = classifier.predict(arr10)
	if y_pred == 1:
		username="Fraud Transaction"
		dark="/true.jpg"
	if y_pred == 0:
		username="Normal Transaction"
		dark="/safe.png"
    		
	return render_template('next.html', data={'des':"Transaction" ,'output1': username, 'output': dark ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})



@app.route('/describe1', methods=['POST'])
def describe1():
	if request.method == 'POST':
		
		return render_template('next.html', data={'des':"Data Description",'out': Od, 'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/dataset', methods=['POST'])
def dataset():
	if request.method == 'POST':
		username=request.form['username']
		return render_template('next.html', data={'des':"dataSet",'output': creditcard,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/d1', methods=['POST'])
def d1():
	if request.method == 'POST':
		Od1="/describe.png"
		creditcard.hist(figsize = (20,20))
		return render_template('next.html', data={'des':"Columns Histogram",'output': Od1,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/d2', methods=['POST'])
def d2():
	if request.method == 'POST':
		Od2="/2d1.png"
		Odi="/2d2.png"
		sns.set_style("whitegrid")
		sns.FacetGrid(creditcard, hue="Class", height = 6).map(plt.scatter, "Time", "Amount").add_legend()
		return render_template('next.html', data={'des':"2D-Scatter Plot",'output': Od2, 'output2': Odi ,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/d3', methods=['POST'])
def d3():
	if request.method == 'POST':
		Od3="/3d.png"
		FilteredData = creditcard[['Time','Amount', 'Class']]
		sns.set_style("whitegrid");
		sns.pairplot(FilteredData, hue="Class", height=5);
		return render_template('next.html', data={'des':"3D-Scatter Plot",'output': Od3,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})

@app.route('/d4', methods=['POST'])
def d4():
	if request.method == 'POST':
		Od4="/1d1.png"
		FilteredData = creditcard[['Time','Amount', 'Class']]
		creditCard_genuine = FilteredData.loc[FilteredData["Class"] == 0]
		creditCard_fraud = FilteredData.loc[FilteredData["Class"] == 1]
		plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o")
		plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o")
		return render_template('next.html', data={'des':"1D-Scatter Plot With Only Amount",'output': Od4,'output3': acc1,'output4':prec,'output5':rec,'output6':MCC,'output7':n_errors})
		

if __name__ == '__main__':
	app.run()











	