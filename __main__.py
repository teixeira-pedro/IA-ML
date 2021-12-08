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
            y_i_B_notB=int(row_i[len(row_i)-2])
            y_i_C_or_D=row_i[len(row_i)-1]
            if y_i_C_or_D == 'None':
                y_i_C_or_D=0
            else :
                y_i_C_or_D=int(y_i_C_or_D)
            y_i_textual=row_i[len(row_i)-4]
            x_i=[]
            for i in range(len(row_i)-4):
                if i != 1:
                    x_i.append(float(row_i[i]))
                else:
                    x_i.append(row_i[i])
            resp.append({'x':x_i,'y':y_i,'y_B_notB':y_i_B_notB,'y_C_or_D':y_i_C_or_D,'yText':y_i_textual})

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
#print(dt_train)
print(i,'/',len(dt))
for i in range(i,len(dt)):
    dt_test.append(dt[i])
for i in dt_train:
    print(i)

XX_train=convert_to_matriz_de_exemplos_X(dt_train,'x')
yy_train=convert_to_matriz_de_exemplos_X(dt_train,'y')
yy_train2=convert_to_matriz_de_exemplos_X(dt_train,'y_B_notB')
yy_train3=convert_to_matriz_de_exemplos_X(dt_train,'y_C_or_D')

XX_test=convert_to_matriz_de_exemplos_X(dt_test,'x')
yy_test=convert_to_matriz_de_exemplos_X(dt_test,'y')
yy_test2=convert_to_matriz_de_exemplos_X(dt_test,'y_B_notB')
yy_test3=convert_to_matriz_de_exemplos_X(dt_test,'y_C_or_D')


LR_A_notA=LogisticRegression()
LR_B_notB=LogisticRegression()
LR_C_or_D=LogisticRegression()
LR_A_notA.fit(XX_train,yy_train)
LR_B_notB.fit(XX_train,yy_train2)
LR_C_or_D.fit(XX_train,yy_train3)

predicoes=[]
predictions_A_notA=LR_A_notA.predict(XX_test)
predictions_B_notB=LR_B_notB.predict(XX_test)
predictions_C_or_D=LR_C_or_D.predict(XX_test)




print('predições do modelo\n',list(predictions_A_notA))
print('y_real\n',yy_test)