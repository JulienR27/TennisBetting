import streamlit as st
#import des principaux packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_style("darkgrid")
from PIL import Image
import webbrowser
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import neighbors

selection = st.sidebar.radio(
     "Selectionnez une partie:",
     ('Introduction', 'Base des cotes', 'DataViz', 
      'Prédiction', 'Stratégies de paris'))
selection = 'Prédiction'
if selection == 'Introduction':
     st.markdown("<h1 style='text-align: center; color: red;'>TennisBetting</h1>", unsafe_allow_html=True)
     image = Image.open('Tennis.jpg')
     st.image(image)
     st.markdown("<h3 style='text-align: left; color: blue;'>Introduction</h3>", unsafe_allow_html=True)
     
     st.write("L'objectif de ce projet était l'étude du pouvoir prédictif des modèles de machine learning dand l'univers du tennis "
              "professionel. Le sujet pouvait être envisagé sous plusieurs formes:"
             )
             
     st.write("1) Améliorer la prédiction des bookmaker en utilisant leurs cotes ainsi que d'autres statistiques.")
     st.write("2) Egaler ou surpasser la prédiction des bookmakers en utilisant que des données de jeu.")
     st.write("3) La création d'un algorithme de paris systématique générant du gain de manière régulière.")
     
     st.write("Nous traiterons ces 3 thèmes dans cette présentation"
             )
     
if selection == 'Base des cotes':
     st.markdown("<h3 style='text-align: center; color: blue;'>La base de cotes</h3>", unsafe_allow_html=True)
     
     st.write("La première base à laquelle nous avons eu accès ne donnait que très peu d'éléments."
              "Elle comportait une liste de match de tennis étiquetés (Winner/Loser), de 2000 à 2018."
              "Néanmoins, elle comportait plusieurs features indispensables :"
              )
              
     st.write("1) Les cotes Winner/Loser de deux bookmakers Pinnacle et B365.")
     st.write("2) Le Classement elo des deux joueurs avant le match.")
     st.write("3) La probabilité elo que le vainqueur gagne le match.")
     
     st.write("Ce sont ces features que avons nettoyés et conservés pour notre étude."
              )
              
     st.markdown("<h4 style='text-align: left; color: green;'>Le classement elo</h4>", unsafe_allow_html=True)
     
     st.write("Le classement elo fut inventé par  Arpad Elo, un professeur de physique et joueur d’échecs américain d’origine hongroise."
              "Il s'agit d'un classement dynamique qui est mis à jour pour chaque joueur après chaque match. Utilisé pour les échecs "
              "à l'origine il est aujourd'hui calculé pour de nombreux autres sports. Vous trouverez une documentation abondante sur internet :"
              )

     url1 = 'https://fr.wikipedia.org/wiki/Classement_Elo'

     if st.button('Wikipedia classement elo'):
        webbrowser.open_new_tab(url1)
        
     url2 = 'https://accromath.uqam.ca/2022/02/le-systeme-de-notation-elo/'
     
     if st.button('accromath classement elo'):
        webbrowser.open_new_tab(url2)
     
     st.markdown("<h4 style='text-align: left; color: green;'>Les cotes</h4>", unsafe_allow_html=True)
     
     st.write("Les cotes winner et loser peuvent être vues comme des inverses de probabilités de victoire."
              " Elles vérifient donc en théorie la formule suivante :"
              )
     st.latex(r'''1 / C_w + 1 / C_l = 1''')
     
     st.write("En réalité les bookmakers ont tendance à réduire légèrement les cotes pour s'assurer sur le "
              "long terme une manne financière constante pour peu que leur précision soit suffisamment élevée."
              " Il en résulte que la somme des inverses des cotes est en général légèrement supérieur à 1."
              )
     st.latex(r'''1 / C_w + 1 / C_l = 1,02''')   
     
     st.markdown("<h4 style='text-align: left; color: green;'>Aperçu de la base</h4>", unsafe_allow_html=True)

     path = "atp_data.csv"
     df = pd.read_csv(path, sep =",",index_col = 0)
     st.dataframe(df)
     
     
if selection == 'DataViz':
     option = st.radio('Que voulez-vous visualiser?',
                         ('distribution des cotes', 'simulation de Monte Carlo', 'cotes moyennes maximales', "le classement elo")) 
    
     if option == 'distribution des cotes':    
         image1 = Image.open('Distribution.jpg')
         st.image(image1)
         image2 = Image.open('StatsCotes.jpg')
         st.image(image2)         
         image3 = Image.open('Corr.jpg')
         st.image(image3)
         
         st.write("On constate plusieurs éléments intéressant :"
              )
              
         st.write("1) Les cotes mediannes Winner se situe autour de 1,5 tandis que les cotes Loser sont autour de 2.4")
         st.write("2) Pinnacle est en moyenne plus attractif que B365 notamment sur les cotes Loser.")
         st.write("3) Pinnacle cote de manière beaucoup plus continue que B365.")
         st.write("4) Les cotes ainsi que la proba elo sont très corréllées.")
         
     if option == 'simulation de Monte Carlo':
         st.write("Nous observons ici le resultat de plusieurs stratégies 'naives' :"
                 )
         st.write("1) Jouer la plus petite cote (favori) de manière systématique.")
         st.write("2) Jouer la grande cote (non favori) de manière systématique.")
         st.write("3) Jouer de manière aléatoire (Monte Carlo).")
         
         image1 = Image.open('MonteCarlo.jpg')
         st.image(image1)
         image2 = Image.open('Convergence.jpg')
         st.image(image2)
         st.write("Dans le cadre d'une simulation de Monte Carlo il est indispensable de "
                  "vérifier la convergence du modèle."
                 )
     if option == 'cotes moyennes maximales':
         st.write("Voyons dans cette partie la cote maximale qu'un bookmaker peut se permettre d'offrir à ses clients"
                  " sans perdre de l'argent en connaissant son accuracy."
                  " Voici nos hypothèses :"
                 )
         st.write("1) Le bookmaker connait son accuracy, par exemple il sait qu'en moyenne x% des favoris en terme de cotes sortent vainqueurs.")
         st.write("2) On restreint l'univers de joueur à ceux qui parient sur le favori.")
         st.write("3) On peut donc estimer l'espérance de gain du bookmaker ainsi:")
         st.latex(r'''E_{gain} = accuracy*(cote_{moyenne}-1) - (1-accuracy)''')
         st.write("4) Pour ne pas perdre d'argent l'espérance de gain doit être supérieure à 0.")
         st.write("4) La cote moyenne limite doit donc vérifier : ")
         st.latex(r'''cote_{moyenne} = ((1-accuracy)/accuracy) + 1''')
         image1 = Image.open('CoteLimite.jpg')
         st.image(image1)
         
     if option == "le classement elo":
     
         path = "atp_data.csv"
         df = pd.read_csv(path, sep =",",index_col = 0)
         
         values = df['Winner'].tolist()
         values = pd.Series(values)
         values = values.value_counts().index
         
         a = st.selectbox('Choose a player', values)
         
         #On récupère le groupe Federer dans l'ensemble winner et looser
         federer_winner = df.groupby("Winner").get_group(a)
         federer_loser = df.groupby("Loser").get_group(a)
         #On ne retient que les colonnes qui nous intéressent
         federer_winner = federer_winner.loc[:,["Date", "WRank", "elo_winner"]]
         federer_loser = federer_loser.loc[:,["Date", "LRank", "elo_loser"]]
         #On renomme les colonnes
         federer_loser = federer_loser.rename(columns={"LRank": "Rank", "elo_loser": "elo"})
         federer_winner = federer_winner.rename(columns={"WRank": "Rank", "elo_winner": "elo"})
         #On concatene winner et looser puis on tri sur les dates
         federer = pd.concat([federer_winner, federer_loser], axis = 0)
         federer = federer.sort_values(by = "Date")
         federer = federer.set_index("Date")
         
         #On réalise un min max scaling pour avoir les valeurs sur la meme échelle
         scaler = MinMaxScaler()
         scaler.fit(federer)
         federer_scaled = pd.DataFrame(scaler.transform(federer), index = federer.index, columns = federer.columns)
         
         st.dataframe(federer)
         fig = plt.figure(figsize = (20,10))
         ax1 = fig.add_subplot(111)

         t = ax1.set_title("Comparaison classement elo et ATP " + a +" (scaled)", fontsize = 14)

         ax1.plot(federer_scaled.index, federer_scaled["Rank"],"c", label = "ATP Rank")
         ax1.plot(federer_scaled.index, federer_scaled["elo"],"red", label = "Classement Elo")

         locs_x, labels_x = plt.xticks()
         locs_y, labels_y = plt.yticks()
         loc_x_new = np.arange(locs_x[0], locs_x[-1], step = 100)
         plt.xticks(loc_x_new)

         plt.xticks(fontsize = 10)
         plt.yticks(fontsize = 10)

         ax1.legend(loc = "best",  fontsize = 12);
         
         st.pyplot(fig)
         
if selection == 'Prédiction':

    df_without_ratios = pd.read_csv('data_with_names_without_ratios_and_bookies2.csv')
    df_without_ratios = df_without_ratios.drop(columns = ["Unnamed: 0"])

    X_train = pd.read_csv('X_train.csv')
    X_train = X_train.drop(columns = ["Unnamed: 0"])

    X_test = pd.read_csv('X_test.csv')
    X_test = X_test.drop(columns = ["Unnamed: 0"])

    y_train = pd.read_csv('y_train.csv')
    y_train = y_train.drop(columns = ["Unnamed: 0"])

    y_test = pd.read_csv('y_test.csv')
    y_test = y_test.drop(columns = ["Unnamed: 0"])

    #st.title("Modélisation")
    st.markdown("<h3 style='text-align: center; color: blue;'>Prédiction</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; color: green;'>Modélisation</h4>", unsafe_allow_html=True)
    #st.write("Sélectionnez un modèle")
    options1 = ["Régression logistique", "KNN", "Arbre de décision", "Random Forest", "XGBoost"]

    choix3 = st.radio("Choisir un modèle :", options=options1)
    if choix3 == options1[0]:
        lr = load('lr_test.joblib')
        st.subheader("Précision : " + str(round(lr.score(X_test, y_test)*100,2)) + '%')
    elif choix3 == options1[1]:
        knn = load('KNN_test.joblib')
        n_neighbors = st.slider('Nombre de voisins', 0, 150, 5)
        knn = neighbors.KNeighborsClassifier(n_neighbors) 
        knn.fit(X_train,y_train)
        st.subheader("Précision : " + str(round(knn.score(X_test, y_test)*100,2)) + '%')
    elif choix3 == options1[2]:
        tree = load('tree_test.joblib')
        st.subheader("Précision : " + str(round(tree.score(X_test, y_test)*100,2)) + '%')
    elif choix3 == options1[3]:
        rf = load('rf_test.joblib')
        st.subheader("Précision : " + str(round(rf.score(X_test, y_test)*100,2)) + '%') 
    elif choix3 == options1[4]:
        xgb1 = load('xgb_test.joblib')
        test = xgb.DMatrix(X_test, y_test)
        xgb1_pred_proba = xgb1.predict(test)
        xgb1_pred = pd.Series([1 if p>0.5 else 0 for p in xgb1_pred_proba])
        st.subheader("Précision : " + str(round(classification_report(y_test, xgb1_pred, output_dict=True)['macro avg']['precision']*100,2)) + '%')
        
    #st.header("Prédictions des matchs de Roland Garros 2018")
    st.markdown("<h4 style='text-align: left; color: green;'>Prédictions des matchs de Roland Garros 2018</h4>", unsafe_allow_html=True)
    tab = ["Première partie", "Deuxième partie", "Troisième partie", "Quatrième partie", "Tableau final"]
    select_tab = st.selectbox("Choisir une partie de tableau à afficher", tab)
    if select_tab == tab[0]:
        image = Image.open('Premiere_partie_RG_2018.PNG')
        st.image(image, caption='Première partie')
    elif select_tab == tab[1]:
        image = Image.open('Deuxieme_partie_RG_2018.PNG')
        st.image(image, caption='Tableau final')
    elif select_tab == tab[2]:
        image = Image.open('Troisieme_partie_RG_2018.PNG')
        st.image(image, caption='Tableau final')
    elif select_tab == tab[3]:
        image = Image.open('Quatrieme_partie_RG_2018.PNG')
        st.image(image, caption='Tableau final')
    elif select_tab == tab[4]:
        image = Image.open('Tableau_final_RG_2018.PNG')
        st.image(image, caption='Tableau final')
            
    choix_joueur1_RG = st.text_input("Choisir le joueur 1")
    choix_joueur2_RG = st.text_input("Choisir le joueur 2")
    #st.write(df[df["p1_name"].str.contains(choix_joueur1_RG)].iloc[-1,:])
    #st.write(df[df["p1_name"].str.contains(choix_joueur2_RG)].iloc[-1, :])
    # index_last_match_p1 = df[df["p1_name"].str.contains(choix_joueur1_RG)].iloc[-1,:].index
    # index_last_match_p2 = df[df["p2_name"].str.contains(choix_joueur2_RG)].iloc[-1,:].index
    
    # x = df[(df["p1_name"].str.contains(choix_joueur1_RG)) & (df["p2_name"].str.contains(choix_joueur2_RG))].groupby("p1_win").count()
    # fig1, ax1 = plt.subplots()
    # # ax1.set_xlabel("Nombre de matchs")
    # # ax1.set_ylabel("Profit & Loss")
    # ax1.pie(x["p1_name"])
    # st.pyplot(fig1) 

    #we create a match with the two players
    # choix_joueur2_RG = "GASQUET"
    # choix_joueur1_RG = "JAZIRI"
    a = df_without_ratios[df_without_ratios["p1_name"].str.contains(choix_joueur2_RG)]
    b = df_without_ratios[df_without_ratios["p2_name"].str.contains(choix_joueur2_RG)]
    data_last_match_p1 = df_without_ratios[df_without_ratios["p1_name"].str.contains(choix_joueur1_RG)].iloc[-1,:]
    data_last_match_p2 = df_without_ratios[df_without_ratios["p2_name"].str.contains(choix_joueur2_RG)].iloc[-1,:]
    p1_name = data_last_match_p1["p1_name"]
    p2_name = data_last_match_p2["p2_name"]
    data_last_match = data_last_match_p1.copy()
    # for feature in features:
    #     data_last_match["p2_"+feature] = data_last_match_p2["p2_"+feature]
    for column in data_last_match.index:
        if "p2_" in column:
            data_last_match[column] = data_last_match_p2[column]
    #we compute the ratios
    features = ['1stWon%',
    '2ndWon%',
    '1stServeEffectiveness',
    'Ret2ServPtsRatio',
    'ServeWon%',
    'ReturnWon%',
    'PtsDominanceRatio',
    'BPConverted%',
    'BPRatio',
    'SetWon%',
    'PtsWon%',
    'Pts2Sets_OP_Ratio',
    'GmsWon%',
    'Pts2Gms_OP_Ratio',
    'Gms2Sets_OP_Ratio',
    'BPWon%',
    'BP_OP_Ratio',
    'BPSaved%',
    'BPSaved_OP_Ratio',
    'BPConverted_OP_Ratio',
    'Ace%',
    'DF%',
    '1stServe%',
    '1stReturnWon%',
    'rank_points',
    'rank',
    'age',
    'ht',
    'elo']
    for feature in features:
      data_last_match[('p1_' + feature)] = data_last_match[('p1_' + feature)] / data_last_match[('p2_' + feature)]
      data_last_match.rename({('p1_' + feature):('ratio_' + feature)}, inplace=True)
      data_last_match.drop(('p2_' + feature), inplace=True)
    data_last_match.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_last_match = data_last_match.dropna()

    data_last_match.drop(["p1_name", "p2_name"], inplace=True)

    scaler = load('scaler_for_new_matches_without_bookies2.joblib')
    data_last_match_num = data_last_match.drop(['p1_hand_R', 'p2_hand_R'])
    data_last_match_cat = data_last_match[['p1_hand_R', 'p2_hand_R']]
    data_last_match_scaled = np.append(scaler.transform(np.array(data_last_match_num).reshape(1,-1)), data_last_match_cat)

    if choix3 == options1[0]:
        lr_new_match = load('lr_for_new_matches_without_bookies2.joblib')
        result_pred = lr_new_match.predict_proba(data_last_match_scaled.reshape(1,-1))
        result = pd.DataFrame(result_pred, index=["Probabilité "+ choix3], columns=[p2_name+" gagne", p1_name+" gagne"])
        st.write(result)
    elif choix3 == options1[1]:
        model_new_match = load('KNN_for_new_matches_without_bookies2.joblib')
        # model_new_match = KNeighborsClassifier(neighbors)
        # model_new_match.fit()
        result_pred = model_new_match.predict_proba(data_last_match_scaled.reshape(1,-1))
        result = pd.DataFrame(result_pred, index=["Probabilité "+ choix3], columns=[p2_name+" gagne", p1_name+" gagne"])
        st.write(result)     
    elif choix3 == options1[2]:
        model_new_match = load('tree_for_new_matches_without_bookies2.joblib')
        result_pred = model_new_match.predict_proba(data_last_match_scaled.reshape(1,-1))
        result = pd.DataFrame(result_pred, index=["Probabilité "+ choix3], columns=[p2_name+" gagne", p1_name+" gagne"])
        st.write(result)
    elif choix3 == options1[3]:
        model_new_match = load('rf_for_new_matches_without_bookies2.joblib')
        result_pred = model_new_match.predict_proba(data_last_match_scaled.reshape(1,-1))
        result = pd.DataFrame(result_pred, index=["Probabilité "+ choix3], columns=[p2_name+" gagne", p1_name+" gagne"])
        st.write(result)     
    elif choix3 == options1[4]:
        xgb_new_match = load('xgb_for_new_matches_without_bookies2.joblib')
        result_pred_xgb = xgb_new_match.predict(xgb.DMatrix(data_last_match_scaled.reshape(1,-1), feature_names=data_last_match.index))
        #result_pred_xgb_p1_p2 = np.append(1-result_pred_xgb)
        result_xgb = pd.DataFrame([[1-result_pred_xgb[0], result_pred_xgb[0]]], index=["Probabilité "+ choix3], columns=[p2_name+" gagne", p1_name+" gagne"])
        st.write(result_xgb)    

    # fig3, ax3 = plt.subplots()
    # ax3.pie(result_pred)
    # st.pyplot(fig3) 
    
if selection == 'Stratégies de paris':
    # df = pd.read_csv('C:/Users/Julien/Documents/EI/Datascientest/Projet Paris Sportifs/DTB_Rolling_Features_ratios_w60_clean.csv')
    # df.drop("Unnamed: 0", axis=1, inplace=True)

    # df_without_ratios = pd.read_csv('C:/Users/Julien/Documents/EI/Datascientest/Projet Paris Sportifs/data_with_names_without_ratios_and_bookies.csv')
    # df_without_ratios = df_without_ratios.drop(columns = ["Unnamed: 0"])

    X_train = pd.read_csv('X_train.csv')
    X_train = X_train.drop(columns = ["Unnamed: 0"])

    y_train = pd.read_csv('y_train.csv')
    y_train = y_train.drop(columns = ["Unnamed: 0"])
   
    st.markdown("<h3 style='text-align: center; color: blue;'>Stratégies de paris</h3>", unsafe_allow_html=True)
    
    options1 = ["Régression logistique", "KNN", "Arbre de décision", "Random Forest", "XGBoost"]

    choix3 = st.radio("Choisir un modèle", options=options1)
    
    strats = ["On parie sur le joueur le mieux coté",
              "On parie sur le joueur qui a une probabilité de gagner supérieure à un certain seuil selon notre modèle",
              "On parie sur le joueur donné gagnant si notre modèle est plus confiant que les bookmakers d'un certain seuil",
              "On parie sur le joueur donné perdant (plus grosse cote) si notre modèle est moins pessimiste que les bookmakers (d'un certain seuil)"]
    
    #choix_strat = strats[0]
    data_scaled = pd.read_csv('data_scaled_from_train.csv').drop("Unnamed: 0", axis=1)
    data_not_scaled = pd.read_csv('data_not_scaled.csv').drop("Unnamed: 0", axis=1)
    target = pd.read_csv('target.csv').drop("Unnamed: 0", axis=1)
    #target = pd.Series([1 if x else 0 for x in target], name='p1_win')

    proba_B365 = 1 / data_not_scaled['p1_B365']
    proba_B365 = pd.concat([proba_B365, 1 / data_not_scaled['p2_B365']], axis=1)
    proba_B365['B365_pred'] = [1 if (row['p1_B365'] > row['p2_B365']) else 0 for i, row in proba_B365.iterrows()]
    proba_PS = 1 / data_not_scaled['p1_PS']
    proba_PS = pd.concat([proba_PS, 1 / data_not_scaled['p2_PS']], axis=1)
    proba_PS['PS_pred'] = [1 if (row['p1_PS'] > row['p2_PS']) else 0 for i, row in proba_PS.iterrows()]

    VSbookmakers_pred = pd.concat([proba_B365, proba_PS, target], axis=1)

    if choix3 == options1[0]:
        lr = load('lr_test.joblib')
        y_pred_proba = pd.DataFrame(lr.predict_proba(data_scaled), index=data_scaled.index) 
        VSbookmakers_pred['model_p1'] = y_pred_proba[1]
        VSbookmakers_pred['model_p2'] = y_pred_proba[0]
    elif choix3 == options1[1]:
        knn = load('KNN_test.joblib')
        n_neighbors = st.slider('Nombre de voisins', 0, 150, 5)
        knn = neighbors.KNeighborsClassifier(n_neighbors) 
        knn.fit(X_train,y_train)
        y_pred_proba = pd.DataFrame(knn.predict_proba(data_scaled), index=data_scaled.index) 
        VSbookmakers_pred['model_p1'] = y_pred_proba[1]
        VSbookmakers_pred['model_p2'] = y_pred_proba[0]
    elif choix3 == options1[2]:
        tree = load('tree_test.joblib')
        y_pred_proba = pd.DataFrame(tree.predict_proba(data_scaled), index=data_scaled.index) 
        VSbookmakers_pred['model_p1'] = y_pred_proba[1]
        VSbookmakers_pred['model_p2'] = y_pred_proba[0]
    elif choix3 == options1[3]:
        rf = load('rf_test.joblib')
        y_pred_proba = pd.DataFrame(rf.predict_proba(data_scaled), index=data_scaled.index) 
        VSbookmakers_pred['model_p1'] = y_pred_proba[1]
        VSbookmakers_pred['model_p2'] = y_pred_proba[0]
    elif choix3 == options1[4]:
        xgb1 = load('xgb_test.joblib')
        y_pred_proba = pd.DataFrame(xgb1.predict(xgb.DMatrix(data_scaled)), index=data_scaled.index) 
        VSbookmakers_pred['model_p1'] = y_pred_proba[0]
        VSbookmakers_pred['model_p2'] = 1 - y_pred_proba[0]
    
    choix_strat = st.radio("Choisir une stratégie", options=strats)
    if choix_strat == strats[0]:
        VSbookmakers_pred["bet_result"] = [(1/row['p1_B365']-1) if ((row['p1_B365'] > row['p2_B365']) and (row['p1_win']==True)) 
                                                               else ((1/row['p2_B365']-1) if ((row['p1_B365'] < row['p2_B365']) and (row['p1_win']==False))
                                                                                             else -1)
                                       for i, row in VSbookmakers_pred.iterrows()]
    elif choix_strat == strats[1]:   
        threshold = st.slider('Seuil %', min_value=50, max_value=100, step=1)
        threshold = threshold / 100        
        #Strategy
        VSbookmakers_pred["bet_result"] = [0 if ((1-threshold) < row['model_p1'] < threshold) 
                                   else ((1/row['p1_B365']-1) if ((row['model_p1']>threshold)&(row['p1_win']==True)) 
                                                                     else ((1/row['p2_B365']-1) if ((row['model_p1']<threshold)&(row['p1_win']==False)) 
                                                                           else -1))
                                     for i, row in VSbookmakers_pred.iterrows()]
    elif choix_strat == strats[2]:   
        threshold = st.slider('Seuil %', 0, 20, 1)
        threshold = threshold / 100
        #Strategy
        VSbookmakers_pred["bet_result"] = [0 if ((abs(row['model_p1']-0.5) - abs(row['p1_B365']-0.5)) < threshold)|((abs(row['model_p2']-0.5) - abs(row['p2_B365']-0.5)) < threshold)
                                   else ((1/row['p1_B365']-1) if (((row['model_p1']-0.5)>0)&(row['p1_win']==True)) 
                                                                     else ((1/row['p2_B365']-1) if (((row['model_p1']-0.5)<0)&(row['p1_win']==False)) 
                                                                           else -1))
                                     for i, row in VSbookmakers_pred.iterrows()]     
    elif choix_strat == strats[3]:   
        threshold = st.slider('Seuil %', 0, 20, 1)
        threshold = threshold / 100
        #Strategy
        VSbookmakers_pred["bet_result"] = [1/row['p1_B365']-1 if ((row['p1_B365'] < 0.5)&(row['p1_B365']+threshold < row['model_p1'])&(row['p1_win']==True))
                                         else ((1/row['p2_B365']-1) if ((row['p2_B365'] < 0.5)&(row['p2_B365']+threshold < row['model_p2'])&(row['p1_win']==False))
                                                                          else (-1 if ((row['p1_B365'] < 0.5)&(row['p1_B365']+threshold < row['model_p1'])&(row['p1_win']==False))|((row['p2_B365'] < 0.5)&(row['p2_B365']+threshold < row['model_p2'])&(row['p1_win']==True))
                                                                               else 0))
                                        for i, row in VSbookmakers_pred.iterrows()]


    VSbookmakers_pred["P&L"] = VSbookmakers_pred["bet_result"].cumsum()
    st.write("Nombre de paris : " + str(VSbookmakers_pred[VSbookmakers_pred["bet_result"]!=0].count()[0]))    
    #♣st.area_chart(VSbookmakers_pred["p&l"])   

    fig2, ax2 = plt.subplots()
    #ax.plot(VSbookmakers_pred["p&l"])
    ax2.set_xlabel("Nombre de matchs")
    ax2.set_ylabel("Profit & Loss")
    ax2.fill_between(VSbookmakers_pred.index, VSbookmakers_pred["P&L"], 0, where = VSbookmakers_pred["P&L"]>0, facecolor="green")
    ax2.fill_between(VSbookmakers_pred.index, VSbookmakers_pred["P&L"], 0, where = VSbookmakers_pred["P&L"]<0, facecolor="red")
    st.pyplot(fig2)        
