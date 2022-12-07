Execute python file:
- open command prompt in this directory (type cmd in windows file explorer address bar)
- execute following code:

	python3 tool.py BPI_Challenge_2012-training.csv BPI_Challenge_2012-test.csv


or with different input parameters:

	python3 tool.py -trainset- -testset-


if a module is not found:

	python3 -m pip install -module-


for example:

	python3 -m pip install xgboost


runtime: ~7 minutes
CPU usage: ~12%


output:
train and test set each with 6 additional columns:
	3 activity_pred columns give prediction of next combination of activity and lifecycle transition in a trace
	they are encoded, see dictionary below for decoding
		activity_pred_baseline: baseline prediction
		activity_pred_RF: Random forest prediction
		activity_pred_RFGS: Random Forest after Grid Search prediction
	
	3 time_pred colums give prediction of time untile next activity in a trace in minutes
		time_pred_baseline: baseline prediction
		time_pred_OLS: multiple linear regression prediction
		time_pred_XGB: XGBoost prediction
	

0: end of trace
1: 'A_ACCEPTEDCOMPLETE'
2: 'A_ACTIVATEDCOMPLETE'
3: 'A_APPROVEDCOMPLETE'
4: 'A_CANCELLEDCOMPLETE'
5: 'A_DECLINEDCOMPLETE'
6: 'A_FINALIZEDCOMPLETE'
7: 'A_PARTLYSUBMITTEDCOMPLETE'
8: 'A_PREACCEPTEDCOMPLETE'
9: 'A_REGISTEREDCOMPLETE'
10:'A_SUBMITTEDCOMPLETE'
11:'O_ACCEPTEDCOMPLETE'
12:'O_CANCELLEDCOMPLETE'
13:'O_CREATEDCOMPLETE',
14:'O_DECLINEDCOMPLETE'
15:'O_SELECTEDCOMPLETE'
16:'O_SENTCOMPLETE'
17:'O_SENT_BACKCOMPLETE'
18:'W_Afhandelen leadsCOMPLETE'
19:'W_Afhandelen leadsSCHEDULE'
20:'W_Afhandelen leadsSTART'
21:'W_Beoordelen fraudeCOMPLETE'
22:'W_Beoordelen fraudeSCHEDULE'
23:'W_Beoordelen fraudeSTART'
24:'W_Completeren aanvraagCOMPLETE'
25:'W_Completeren aanvraagSCHEDULE'
26:'W_Completeren aanvraagSTART'
27:'W_Nabellen incomplete dossiersCOMPLETE'
28:'W_Nabellen incomplete dossiersSCHEDULE'
29:'W_Nabellen incomplete dossiersSTART'
30:'W_Nabellen offertesCOMPLETE'
31:'W_Nabellen offertesSCHEDULE'
32:'W_Nabellen offertesSTART'
33:'W_Valideren aanvraagCOMPLETE'
34:'W_Valideren aanvraagSCHEDULE'
35:'W_Valideren aanvraagSTART'

