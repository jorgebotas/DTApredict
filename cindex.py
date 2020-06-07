import pandas as pd







def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])


    if pair is not 0:
        return summ/pair
    else:
        return 0

preds = pd.read_csv("cindex/bdb_predictions.csv")

exp = preds.Experimental
pred = preds.Predicted

print(exp)
print(pred)

cindex = get_cindex(exp, pred)

print(cindex)
