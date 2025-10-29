import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Oorspronkelijke koers")

tab1, tab2, tab3 = st.tabs(["Verkenning","Analyse","Model"])

with tab1:
    st.header("Data verkenning")
    st.write(
        "In dit onderdeel wordt de dataset onderzocht om inzicht te krijgen in de structuur, "
        "ontbrekende waarden en verdeling van variabelen. "
        "Daarnaast worden nieuwe kenmerken toegevoegd die later gebruikt kunnen worden voor het voorspellende model."
    )
    st.markdown("---")

    train = pd.read_csv("bestanden/train.csv")
    st.dataframe(train.head())   
    st.markdown("---")

    st.write("Missing values")
    st.write(train.isna().sum())
    st.write("Er is gekozen om de 'Cabin' kolom te droppen. De 'Embarked' kolom wordt aangevuld met de modus en de \
             'Age' kolom blijft zoals deze is")
    
    train['Embarked'] =  train['Embarked'].fillna(train['Embarked'].mode()[0])
    train = train.drop(columns='Cabin')

    st.write("De dataframe ziet er vervolgens zo uit:")
    st.write(train.isna().sum())

    st.write("De describe:")
    st.dataframe(train.describe(include='all'))

with tab2:
    st.header("Data analyse")
    st.write(
        "Analyse van patronen, relaties en variabelen die invloed hebben op de overlevingskans, "
        "met behulp van grafieken en kruistabellen."
    )

    st.write("Data analyse")
    sns.set_style('whitegrid')
    sns.set_palette('deep')

    # 1. Overlevingspercentage per geslacht
    plt.figure()
    sns.barplot(data=train, x='Sex', y='Survived')
    plt.title('Overlevingspercentage per geslacht')
    plt.xlabel('Geslacht')
    plt.ylabel('Gemiddelde overlevingspercentage')
    st.pyplot(plt)

    # 2. Overlevingskans per klasse
    plt.figure()
    sns.barplot(data=train, x='Pclass', y='Survived')
    plt.title('Overlevingskans per class')
    plt.xlabel('Passagiersklasse')
    plt.ylabel('Gemiddelde overlevingspercentage')
    st.pyplot(plt)

    # 3. Leeftijdsverdeling
    plt.figure()
    sns.histplot(data=train['Age'], bins=20)
    plt.title('Leeftijdsverdeling')
    plt.xlabel('Leeftijd')
    plt.ylabel('Aantal passagiers')
    st.pyplot(plt)

    # 4. Leeftijdsverdeling per geslacht en overleving
    plt.figure()
    sns.boxplot(data=train, x='Sex', y='Age', hue='Survived')
    plt.title("Leeftijdsverdeling per geslacht en overleving")
    plt.xlabel('Geslacht')
    plt.xticks(ticks=[0,1], labels=['Man','Vrouw'])
    plt.ylabel('Leeftijd')
    plt.legend(title='Overleefd')
    st.pyplot(plt)

    # 5. Leeftijd vs Ticketprijs per overlevingsstatus
    plt.figure()
    sns.scatterplot(data=train, x='Age', y='Fare', hue='Survived')
    plt.title("Leeftijd vs Ticketprijs per overlevingsstatus")
    plt.xlabel("Leeftijd")
    plt.ylabel("Ticketprijs")
    plt.legend(title='Overleefd')
    st.pyplot(plt)

    # 6. Verdeling van leeftijden per overlevingsstatus
    plt.figure()
    sns.histplot(data=train, x='Age', hue='Survived', multiple='stack')
    plt.title("Verdeling van leeftijden per overlevingsstatus")
    plt.xlabel('Leeftijd')
    plt.ylabel('Aantal passagiers')
    plt.annotate('Veel overlevende \n jonge kinderen', xy=(3, 25), xytext=(1, 60),
                arrowprops=dict(color='red', arrowstyle='->', lw=2))
    plt.legend(title='Overleefd', labels=['Ja', 'Nee'])
    st.pyplot(plt)

        # Crosstabs berekenen
    sexsurvived = pd.crosstab(train['Sex'], train['Survived'], normalize='index')
    classsurvived = pd.crosstab(train['Pclass'], train['Survived'], normalize='index')
    embarkedsurvived = pd.crosstab(train['Embarked'], train['Survived'], normalize='index')

    train['AgeMissing'] = train['Age'].isna()
    agemissingsurvived = pd.crosstab(train['AgeMissing'], train['Survived'], normalize='index')

    sexclasssurvived = pd.crosstab([train['Sex'], train['Pclass']], train['Survived'], normalize='index')
    st.markdown("---")
    st.header('Crosstabs')

    # Weergave in Streamlit
    st.subheader("Overlevingspercentage per geslacht")
    st.dataframe(sexsurvived)

    st.subheader("Overlevingspercentage per klasse")
    st.dataframe(classsurvived)

    st.subheader("Overlevingspercentage per opstaplocatie")
    st.dataframe(embarkedsurvived)

    st.subheader("Overlevingspercentage bij ontbrekende leeftijdsdata")
    st.dataframe(agemissingsurvived)

    st.subheader("Overlevingspercentage per geslacht en klasse gecombineerd")
    st.dataframe(sexclasssurvived)

    # ===== 1. Pclass verdeling =====
    st.subheader("Verdeling passagiers per klasse")
    st.write(train['Pclass'].value_counts(normalize=True))  # 'index' argument mag weg bij Series

    # ===== 2. Titel uit naam halen =====
    def get_title(Name):
        splitkomma = Name.split(',')[1]
        splitpunt = splitkomma.split('.')[0]
        Title = splitpunt.strip()
        return Title

    train['Title'] = train['Name'].apply(get_title)

    st.subheader("Overlevingspercentage per titel")
    st.dataframe(pd.crosstab(train['Title'], train['Survived'], normalize='index'))

    # ===== 3. Ontbrekende leeftijd =====
    train['AgeMissing'] = train['Age'].isna()
    st.subheader("Overlevingspercentage bij ontbrekende leeftijdsdata")
    st.dataframe(pd.crosstab(train['AgeMissing'], train['Survived'], normalize='index'))

    # ===== 4. Leeftijdsgroepen =====
    train['AgeGroup'] = pd.cut(train['Age'].dropna(), bins=[0, 12, 60, 100],
                            labels=['Child(0-12)', 'Adult(12-60)', 'Old(60+)'])
    agesurvived = pd.crosstab(train['AgeGroup'], train['Survived'], normalize='index')

    st.subheader("Overlevingspercentage per leeftijdsgroep")
    st.dataframe(agesurvived)

    # ===== 5. Familieomvang =====
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    train['IsAlone'] = train['FamilySize'] == 1

    st.subheader("Overlevingspercentage per familieomvang")
    st.dataframe(pd.crosstab(train['FamilySize'], train['Survived'], normalize='index'))

    st.subheader("Overlevingspercentage: alleenreizend vs niet")
    st.dataframe(pd.crosstab(train['IsAlone'], train['Survived'], normalize='index'))

    # ===== 6. Combinatie Crosstab (Geslacht, Klasse, Leeftijdsgroep) =====
    ct = pd.crosstab([train['Sex'], train['Pclass'], train['AgeGroup']],
                    train['Survived'], normalize='index')

    groepgroote = train.groupby(['Sex', 'Pclass', 'AgeGroup']).size()
    filtered_index = groepgroote[groepgroote > 5].index

    st.subheader("Overlevingspercentage per geslacht, klasse en leeftijdsgroep (minstens 5 personen)")
    st.dataframe(ct.loc[filtered_index])

    # eventueel ook aantallen tonen
    st.caption("Aantal passagiers per groep (alleen >5 getoond)")
    st.dataframe(groepgroote[groepgroote > 5])

with tab3:
    st.header("Model")
    st.write(
        "Het gebruikte model, de regels en de behaalde Kaggle-score."
    )
    st.markdown("---")

    st.markdown("""
    Het gekozen model is uitgebreid ten opzichte van het vorige model:

    - Vrouwen uit de 1e en 2e klas jonger dan 60 jaar overleven het  
    - Mannen uit de 1e of 2e klas jonger dan 12 jaar overleven het
    """)

    st.markdown("Met deze submission wordt een Kaggle-score van **0.775** behaald.")
    st.markdown("---")
    