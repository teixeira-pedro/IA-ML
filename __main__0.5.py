__name__='__main__'
from csv import reader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from copy import deepcopy
from random import randint



def split_aleat(XY,k):
    assert(len(XY)>=k and k>1)
    grupos=[]
    for i in range(k):
        grupos.append([])
    XX=deepcopy(XY)
    grupo_atual=0
    while 1:
        if XX :
            i=randint(0,len(XX)-1)
            linha=XX.pop(i)
            if grupo_atual == k:
                grupo_atual=0
            grupos[grupo_atual].append(linha)
            grupo_atual=grupo_atual+1
        else:
            break
    return(grupos)

def valid_cruzada_exemplo(dataset,k):
    grupos=split_aleat(dataset,k)
    ACURACIAS=[]
    for dt in grupos:
        dt_train, dt_test = train_test_split(dt, test_size=0.3, random_state=1)


        LR_A_notA = LogisticRegression(multi_class='multinomial')
        LR_B_notB = LogisticRegression(multi_class='multinomial')
        LR_C_or_D = LogisticRegression(multi_class='multinomial')

        y_A_notA_real = convert_to_matriz_de_exemplos_X(dt_test, 'y')
        y_B_notB_real = convert_to_matriz_de_exemplos_X(dt_test, 'y_B_notB')
        y_C_or_D_real = convert_to_matriz_de_exemplos_X(dt_test, 'y_C_or_D')
        X_test = convert_to_matriz_de_exemplos_X(dt_test, 'x')

        y_A_notA = convert_to_matriz_de_exemplos_X(dt_train, 'y')
        y_B_notB = convert_to_matriz_de_exemplos_X(dt_train, 'y_B_notB')
        y_C_or_D = convert_to_matriz_de_exemplos_X(dt_train, 'y_C_or_D')
        X_train = convert_to_matriz_de_exemplos_X(dt_train, 'x')

        y_real_textual = convert_to_matriz_de_exemplos_X(dt_test, 'yText')

        LR_A_notA.fit(X_train, y_A_notA)
        LR_B_notB.fit(X_train, y_B_notB)
        LR_C_or_D.fit(X_train, y_C_or_D)

        predicoes_A_notA = list(LR_A_notA.predict(X_test))
        predicoes_B_notB = list(LR_B_notB.predict(X_test))
        predicoes_C_or_D = list(LR_C_or_D.predict(X_test))
        predicoes = []
        for i in range(len(predicoes_C_or_D)):
            if predicoes_A_notA[i] == 1:
                predicoes.append('A')
            elif predicoes_B_notB[i] == 1:
                predicoes.append('B')
            elif predicoes_C_or_D[i] == 1:
                predicoes.append('C')
            else:
                predicoes.append('D')


        ACURACIAS.append(acuracia(y_real_textual,predicoes))

    return sum(ACURACIAS)/len(grupos)


def T_(v):
    assert type([])==type(v) and len(v) > 0
    try :
        a=len(v[0])
        resp =[]
        for i in v:
            resp.append(i[0])
        return resp
    except :
        v_t =[]
        for i in v:
            v_t.append([i])
        return v_t

def append_Matriz_vetor_transposto(X,v):
    assert(len(X)==len(v)) #assegurando msm n de linhas
    v_t=v
    X_v_t=[]
    for i in range(len(X)):
        X_v_t.append(X[i]+v_t[i])
    return X_v_t

def acuracia(a,b):
    assert(len(a)==len(b) and len(a)!=0)
    acertos=0
    for i in range(len(a)):
        acertos = acertos + int(a[i]==b[i])
    return acertos/len(a)




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
    return resp


def get_data(arq):
    resp = []
    with open(arq,'r') as FP:
        csv_reader = reader(FP)
        rows=list(csv_reader)
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
    return resp


#==================IMPORTANDO O ARQUIVO, TRANSFORMANDO EM MATRIZ E SEPARANDO O GRUPO DE TESTE E DE TREINO============
dt=convert_M_F_to_0_1(get_data_bin_classe_A_notA('C:\\Users\\Public\\iaml\\csv_tratado_binario.csv'),'x',1)
dt_train,dt_test=train_test_split(dt,test_size=0.3,random_state=1)
print(len(dt_train),'/',len(dt))

#==================IMPORTANDO O ARQUIVO, TRANSFORMANDO EM MATRIZ E SEPARANDO O GRUPO DE TESTE E DE TREINO============


# Apagar os warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


#====================INSTANCIAS LR======================
LR_A_notA=LogisticRegression(multi_class='multinomial')
LR_B_notB=LogisticRegression(multi_class='multinomial')
LR_C_or_D=LogisticRegression(multi_class='multinomial')
#====================INSTANCIAS LR======================





# ====================VETORES Y DE TESTES S MATRIZ DE TESTES======================
y_A_notA_real = convert_to_matriz_de_exemplos_X(dt_test, 'y')
y_B_notB_real = convert_to_matriz_de_exemplos_X(dt_test, 'y_B_notB')
y_C_or_D_real = convert_to_matriz_de_exemplos_X(dt_test, 'y_C_or_D')
X_test = convert_to_matriz_de_exemplos_X(dt_test, 'x')
# ====================VETORES Y DE TESTES S MATRIZ DE TESTES======================


# ====================VETORES Y DE EXEMPLOS S MATRIZ DE EXEMPLOS======================
y_A_notA = convert_to_matriz_de_exemplos_X(dt_train, 'y')
y_B_notB = convert_to_matriz_de_exemplos_X(dt_train, 'y_B_notB')
y_C_or_D = convert_to_matriz_de_exemplos_X(dt_train, 'y_C_or_D')
X_train = convert_to_matriz_de_exemplos_X(dt_train, 'x')
#X_AnotA_train = append_Matriz_vetor_transposto(X_train, T_(y_A_notA))
#X_BnotB_train = append_Matriz_vetor_transposto(X_train, T_(y_B_notB))
#X_CorD_train = append_Matriz_vetor_transposto(X_train, T_(y_C_or_D))
# ====================VETORES Y DE EXEMPLOS S MATRIZ DE EXEMPLOS======================


y_real_textual=convert_to_matriz_de_exemplos_X(dt_test,'yText')

#====================TREINANDO======================
LR_A_notA.fit(X_train,y_A_notA)
LR_B_notB.fit(X_train,y_B_notB)
LR_C_or_D.fit(X_train,y_C_or_D)
#====================TREINANDO======================


#==================CLASSIFICANDO - PENEIRA DE CLASSIFICADORES=================
predicoes_A_notA=list(LR_A_notA.predict(X_test))
predicoes_B_notB=list(LR_B_notB.predict(X_test))
predicoes_C_or_D=list(LR_C_or_D.predict(X_test))
predicoes=[]
print(len(predicoes_A_notA)==len(predicoes_B_notB)==len(predicoes_C_or_D))
for i in range(len(predicoes_C_or_D)):
    if predicoes_A_notA[i]==1:
        predicoes.append('A')
    elif predicoes_B_notB[i]==1:
        predicoes.append('B')
    elif predicoes_C_or_D[i]==1:
        predicoes.append('C')
    else:
        predicoes.append('D')
#==================CLASSIFICANDO - PENEIRA DE CLASSIFICADORES=================

print(y_real_textual)
print(predicoes)

#=======================MÉTRICAS===========================
#Acurácia dos classificadores
print('Acurácias')
print('Acurácia do classificador A ou ¬A',LR_A_notA.score(X_test,y_A_notA_real))
print('Acurácia do classificador B ou ¬B',LR_B_notB.score(X_test,y_B_notB_real))
print('Acurácia do classificador C ou D',LR_C_or_D.score(X_test,y_C_or_D_real))

print('Acurácia geral',acuracia(y_real_textual,predicoes))

#Validação Cruzada
print('\nValidação Cruzada')
mediasA_notA=cross_val_score(LR_A_notA,X_test,y_A_notA_real,cv=5)
mediasB_notB=cross_val_score(LR_B_notB,X_test,y_B_notB_real,cv=5)
mediasC_or_D=cross_val_score(LR_C_or_D,X_test,y_C_or_D_real,cv=5)
print('Validação cruzada para o modelo LR - Classificador A ou ¬A :',sum(mediasA_notA)/len(mediasA_notA))
print('Validação cruzada para o modelo LR - Classificador B ou ¬B :',sum(mediasB_notB)/len(mediasB_notB))
print('Validação cruzada para o modelo LR - Classificador C ou D :',sum(mediasC_or_D)/len(mediasC_or_D))
print('Validação cruzada geral',valid_cruzada_exemplo(dt,5))
#=======================MÉTRICAS===========================



