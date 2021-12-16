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


def convert_to_matriz_de_exemplos_X_2(vals,*keys):
    VALSX=[]
    for i in vals:
        VALSY=[]
        for key in keys:
            VALSY.append(i[key])
        VALSX.append(VALSY)
    return VALSX


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

#====================INSTANCIAS LR======================
LR_A_notA=LogisticRegression()
LR_B_notB=LogisticRegression()
LR_C_or_D=LogisticRegression()
#====================INSTANCIAS LR======================
#====================VETORES Y DE EXEMPLOS S MATRIZ DE EXEMPLOS======================
y_A_notA=convert_to_matriz_de_exemplos_X(dt_train,'y')
y_B_notB=convert_to_matriz_de_exemplos_X(dt_train,'y_B_notB')
y_C_or_D=convert_to_matriz_de_exemplos_X(dt_train,'y_C_or_D')
X=convert_to_matriz_de_exemplos_X(dt_train,'x')
#====================VETORES Y DE EXEMPLOS S MATRIZ DE EXEMPLOS======================
#====================VETORES Y DE TESTES S MATRIZ DE TESTES======================
y_A_notA_real=convert_to_matriz_de_exemplos_X(dt_test,'y')
y_B_notB_real=convert_to_matriz_de_exemplos_X(dt_test,'y_B_notB')
y_C_or_D_real=convert_to_matriz_de_exemplos_X(dt_test,'y_C_or_D')
X_test=convert_to_matriz_de_exemplos_X(dt_test,'x')
#====================VETORES Y DE TESTES S MATRIZ DE TESTES======================
XX_train=[[]]
yy_train=[]
yy_train2=[]
yy_train3=[]

XX_test=convert_to_matriz_de_exemplos_X(dt_test,'x')
yy_test=convert_to_matriz_de_exemplos_X(dt_test,'y')
yy_test2=convert_to_matriz_de_exemplos_X(dt_test,'y_B_notB')
yy_test3=convert_to_matriz_de_exemplos_X(dt_test,'y_C_or_D')

#====================TREINANDO======================
LR_A_notA.fit(X,y_A_notA)
LR_B_notB.fit(X,y_B_notB)
LR_C_or_D.fit(X,y_C_or_D)
#====================TREINANDO======================


#==================CLASSIFICANDO - PENEIRA DE CLASSIFICADORES=================
predicoes_A_notA=LR_A_notA.predict(X_test)
#guardarão os indices de quem é de cada classe dentro da matriz X
A=[]
B=[]
C=[]
D=[]
#guarda quem pertence ao resto
notA=[]
notB=[]


#incio da peneira de classificadores
#-------------------Peneira A ou não A
for i in range(len(predicoes_A_notA)):
    if predicoes_A_notA[i]==0: #NÃO É Classe A
        notA.append(i)
    else: #É CLASSE A, adicionar ao grupo
        A.append(i)
#print('predicao',list(predicoes_A_notA)[:20])
#print('index A',list(A)[:20])
#print('index ¬A',list(notA)[:20])

#-----------------Peneira B ou não B------------------
X_TEMP=[]
for i in notA:
    X_TEMP.append(X[i])
#carregando o novo vetor X pra usar no proximo classificador, já que podemos desconsiderar quem já é A
predicoes_B_notB=LR_B_notB.predict(X_TEMP)
# cada predição p_i corresponde a posição marcado por cada valor do vetor notA
# p_i é previsão de....        vetor notA
#  1                    X[1]       1
#  0                    X[3]       3
#  0                    X[5]       5
#  ...
print(len(notA),len(X_TEMP),len(predicoes_B_notB))
for i in range(len(predicoes_B_notB)):
    if predicoes_B_notB[i]==0: #NÃO É Classe B
        #print(i,notA[i])
        notB.append(notA[i])
    else: #É CLASSE B, adicionar ao grupo
        B.append(notA[i])

#print(len(predicoes_B_notB),len(B),len(notB),len(B)+len(notB),len(X_TEMP))
#print(len(notA),len(X_TEMP))
# -------------------Peneira C ou D------------------
X_TEMP=[]
for i in notB:
    X_TEMP.append(X[i])
predicoes_C_or_D=LR_C_or_D.predict(X_TEMP)
for i in range(len(predicoes_C_or_D)):
    if predicoes_C_or_D[i]==0: #É D
        #print(i,notA[i])
        D.append(notB[i])
    else: #É CLASSE C, adicionar ao grupo
        C.append(notB[i])
#print(len(A),len(B),len(C),len(D),len(A)+len(B)+len(C)+len(D),len(X_test))
#==================CLASSIFICANDO - PENEIRA DE CLASSIFICADORES=================


