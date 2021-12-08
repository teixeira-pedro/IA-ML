__name__='__main__'
from csv import reader
from sklearn.linear_model import LogisticRegression

def convert_M_F_to_0_1(vals,key,campo_MF):
    resp=[]
    for v in vals:
        nv=v
        if nv[key][campo_MF]=='M':
            nv[key][campo_MF]=0
        elif nv[key][campo_MF]=='F':
            nv[key][campo_MF] = 1
        resp.append(nv)
    return resp


def convert_to_matriz_de_exemplos_X(vals,key):
    VALSX=[]
    for i in vals:
        VALSX.append(i[key])
    return VALSX

def get_data_bin_classe_A_notA(arq):
    resp = []
    with open(arq,'r') as FP:
        csv_reader = reader(FP)
        rows=list(csv_reader)
        #print(rows)
    for k in range(len(rows)):
        if k!=0:
            row_i=rows[k]
            y_i=int(row_i[len(row_i)-3])
            x_i=[]
            for i in range(len(row_i)-4):
                if i != 1:
                    x_i.append(float(row_i[i]))
                else:
                    x_i.append(row_i[i])
            resp.append({'x':x_i,'y':y_i})

            #print(x_i,y_i)
    return resp


def get_data(arq):
    resp = []
    with open(arq,'r') as FP:
        csv_reader = reader(FP)
        rows=list(csv_reader)
        #print(rows)
    for k in range(len(rows)):
        if k!=0:
            row_i=rows[k]
            y_i=row_i[len(row_i)-1]
            x_i=[]
            for i in range(len(row_i)-1):
                if i != 1:
                    x_i.append(float(row_i[i]))
                else:
                    x_i.append(row_i[i])
            resp.append({'x':x_i,'y':y_i})

            #print(x_i,y_i)
    return resp

dt=convert_M_F_to_0_1(get_data_bin_classe_A_notA('C:\\Users\\Public\\iaml\\csv_tratado_binario.csv'),'x',1)
dt_train=[]
dt_test=[]
for i in range(int(0.7*len(dt))):
    dt_train.append(dt[i])
print(dt_train)
print(i,'/',len(dt))
for i in range(i,len(dt)):
    dt_test.append(dt[i])

XX_test=convert_to_matriz_de_exemplos_X(dt_train,'x')
yy_test=convert_to_matriz_de_exemplos_X(dt_train,'y')

LR=LogisticRegression()
LR.fit(XX_test,yy_test)
