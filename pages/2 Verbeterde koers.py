import streamlit as st 
import matplotlib.pyplot as plt
import plotly.express as px 
import pandas as pd
import numpy as np
from io import StringIO

st.title("Verbeterde koers")
df = pd.read_csv("bestanden/train.csv")

tab1, tab2, tab3 = st.tabs(["Verkenning","Analyse","Model"])

with tab1:
    st.header("Data verkenning")
    st.write(
        "In dit onderdeel wordt de dataset onderzocht om inzicht te krijgen in de structuur, "
        "ontbrekende waarden en verdeling van variabelen. "
        "Daarnaast worden nieuwe kenmerken toegevoegd die later gebruikt kunnen worden voor het voorspellende model."
    )
    st.markdown("---")

    kleur_stijl = "#ED796C"  
    kleuren_survived = ['#ED796C', "#52C883"] 

    st.dataframe(df.head())

    missing_cols = df.columns[df.isnull().any()]
    missing_count = df[missing_cols].isnull().sum()

    st.markdown(
        "<h3 style='color:#4CAF50;'>Data opschonen</h3>",
        unsafe_allow_html=True
    )
    st.write("Ontbrekende data per kolom: ")
    st.dataframe(missing_count.to_frame("Aantal ontbrekend"))
    st.write(f"Totaal {len(missing_cols)} kolommen met ontbrekende waarden.")


    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df =  df.drop(columns='Cabin')

    st.info(
        "De kolom *Cabin* wordt verwijderd omdat deze te veel ontbrekende waarden bevat. "
        "De kolom *Embarked* wordt opgevuld met de modus (*S*), en *Age* met de mediaan (28)."
    )

    st.write("Nieuwe dataframe:")
    #title
    def get_title(Name):
        splitkomma = Name.split(',')[1]
        splitpunt = splitkomma.split('.')[0]
        Title = splitpunt.strip()
        return Title
    df['Title'] = df['Name'].apply(get_title)
    title_counts = df['Title'].value_counts()
    other_titles = title_counts[title_counts < 10].index
    df['Title'] = df['Title'].replace(other_titles, 'Other')

    #naamlengte
    df['NameLength'] = df['Name'].str.len()
    df['NameLength'] = pd.cut(df['NameLength'], bins=[0, 25, 35, 50, 85], labels=['VeryShort','Short','Medium','Long'], include_lowest=True)

    #familysize en is alone
    df['FamilySize'] = df['SibSp'] +  df['Parch'] + 1
    df['FamilySizeGroup'] = pd.cut(df['FamilySize'], bins=(0,1,4,11), labels=['Small','Medium','Big'])
    df['IsAlone'] = df['FamilySize'] == 1

    #agegroup
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 60, 100], labels=['Child', 'Teen', 'Adult', 'Old(60+)'])

    st.dataframe(df.head())

    st.info(
        "Kolommen die zijn toegevoegd: *IsAlone*, *AgeGroup*, *FamilySize*, "
        "*FamilySizeGroup*, *NameLength* en *Title*."
    )

    st.markdown("---")

    df_num = df.copy()

    # Kolommen met categorieën omzetten naar numeriek
    cat_cols = ['Sex', 'AgeGroup', 'FamilySizeGroup', 'NameLength', 'Title', 'Embarked', 'IsAlone']

    for col in cat_cols:
        df_num[col] = df_num[col].astype('category').cat.codes

    df_num = df_num.drop(columns=['Name','Ticket','SibSp','Parch','PassengerId','FamilySize','Age'])
    title_means = df.groupby('Title')['Survived'].mean()

    title_map = title_means.rank(method='first', ascending=False).astype(int) - 1
    df_num['Title'] = df['Title'].map(title_map)

    corr_survived = df_num.corr(numeric_only=True)[['Survived']].drop('Survived')
    corr_survived['AbsCorr'] = corr_survived['Survived'].abs()

    corr_survived = corr_survived.sort_values('AbsCorr', ascending=False)

    fig = px.imshow(
        corr_survived[['AbsCorr']],
        text_auto=True,
        color_continuous_scale='Reds',    
        title='Sterkte van invloed op overleving (Titanic)',
        aspect='auto'
    )

    fig.update_layout(
        width=400,
        height=600,
        coloraxis_colorbar=dict(title='Sterkte'),
        font=dict(size=12)
    )
    st.markdown(
        "<h3 style='color:#4CAF50;'>Correlatie matrix</h3>",
        unsafe_allow_html=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "De matrix laat zien hoe sterk de verschillende kolommen invloed hebben op de overlevingskans. "
        "Dat de kolom *Title* belangrijk is, valt te verklaren doordat de titels *Miss* en *Mrs* aan vrouwen werden gegeven — "
        "en vrouwen overleefden doorgaans vaker."
    )

    st.markdown("---")

    st.markdown(
        "<h3 style='color:#4CAF50;'>Verdeling per kolom</h3>",
        unsafe_allow_html=True
    )


    kolommen = [
        'Sex',
        'Pclass',
        'Embarked',
        'AgeGroup',
        'FamilySizeGroup',
        'Title',
        'IsAlone',
        'NameLength',
        'Fare'
    ]
    candidate_cols = [c for c in kolommen if c in df.columns]

    col1, col2 = st.columns([3, 1])
    with col1:
        kolom = st.selectbox("Kies een kolom", candidate_cols)
    with col2:
        kleur = st.checkbox("Kleur op Survived", value=False)

    def plot_distribution(df, col, color_survived=False):
        d = df.copy()
        d[col] = d[col].astype(str).fillna("Onbekend")

        if color_survived:
            fig = px.histogram(
                d,
                x=col,
                color='Survived',
                color_discrete_sequence=kleuren_survived,
                barmode='group',
                text_auto=True,
                title=f"Verdeling van {col} per overlevingsstatus"
            )
        else:
            grp = d[col].value_counts().reset_index()
            grp.columns = [col, 'Aantal']
            fig = px.bar(
                grp,
                x=col,
                y='Aantal',
                text='Aantal',
                title=f"Verdeling van {col}",
                color_discrete_sequence=[kleur_stijl]
            )

        fig.update_layout(
            height=450,
            margin=dict(t=60, r=10, l=10, b=10),
            xaxis_title=col,
            yaxis_title="Aantal passagiers",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    def show_basic_info(df, col):
        d = df[col]
        total = len(d)
        missing = d.isna().sum()
        unique = d.nunique()
        mode_val = d.mode()[0] if not d.mode().empty else "—"
        mode_pct = d.value_counts(normalize=True).iloc[0] * 100 if not d.value_counts().empty else 0
        
        st.markdown("### Verkennen")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Aantal", f"{total}")
        
        share_second = d.value_counts(normalize=True).iloc[1] * 100 if unique > 1 else 0
        dominantie = mode_pct - share_second
        c2.metric("Dominantie", f"{dominantie:.1f}%")
        
        c3.metric("Unieke categorieën", f"{unique}")
        c4.metric(
            label="Meest voorkomend",
            value=f"{mode_val}",
            delta=f"{mode_pct:.1f}%",
            delta_color="off"
        )

    plot_distribution(df, kolom, color_survived=kleur)
    show_basic_info(df, kolom)

    st.markdown("---")

with tab2:
    st.header("Data analyse")
    st.write(
        "Analyse van patronen, relaties en variabelen die invloed hebben op de overlevingskans, "
        "met behulp van grafieken en kruistabellen."
    )
    st.markdown("---")

    st.markdown(
        "<h3 style='color:#4CAF50;'>Overlevingskans per kolom</h3>",
        unsafe_allow_html=True
    )
    kolommen = [
    'Sex',
    'Pclass',
    'Embarked',
    'AgeGroup',
    'FamilySizeGroup',
    'Title',
    'IsAlone',
    'NameLength'
    ]

    kleuren = {
        'Sex': '#4C72B0',
        'Pclass': '#55A868',
        'Embarked': "#DA47C4",
        'AgeGroup': '#8172B3',
        'FamilySizeGroup': '#CCB974',
        'Title': '#64B5CD',
        'IsAlone': "#B85252",
        'NameLength':"#23B48D"
    }
    candidate_cols = [c for c in kolommen if c in df.columns]

    cols_sel = st.multiselect(
        "Kies één of meer kolommen",
        options=candidate_cols,
        default=[c for c in ['Sex', 'Pclass','Embarked','AgeGroup','FamilySizeGroup','Title','IsAlone','NameLength'] if c in candidate_cols])

    def plot_survival_by_column(data: pd.DataFrame, col: str):
        d = data[[col, 'Survived']].dropna().copy()
        d['Survived'] = d['Survived'].astype(int)

        grp = (
            d.groupby(col)
            .agg(survival_rate=('Survived', 'mean'), count=('Survived', 'size'))
            .reset_index()
            .sort_values('survival_rate', ascending=False)
        )

        kleur = [kleuren.get(col, '#1f77b4')]
        fig = px.bar(
            grp,
            x=col,
            y='survival_rate',
            text='survival_rate',
            hover_data={'count': True, 'survival_rate': ':.2%'},
            title=f"{col} → overlevingspercentage",
            color_discrete_sequence=kleur
        )
        fig.update_traces(texttemplate="%{text:.0%}", textposition="inside")
        fig.update_yaxes(title="Overlevingspercentage", tickformat=".0%", range=[0, 1])
        fig.update_layout(height=380, margin=dict(t=60, r=10, l=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if not cols_sel:
        st.info("Kies minimaal één kolom hierboven.")
    else:
        for i in range(0, len(cols_sel), 2):
            c1, c2 = st.columns(2)
            with c1:
                plot_survival_by_column(df, cols_sel[i])
            if i + 1 < len(cols_sel):
                with c2:
                    plot_survival_by_column(df, cols_sel[i + 1])

    st.info(
        "Je overlevingskans is aanzienlijk groter als vrouw zijnde. "
        "Ook de klassen zijn gerangschikt op overlevingskans. "
        "Een familie van drie à vier personen heeft de beste kans, en niet alleen reizen had invloed — ook waar je aan boord ging speelde mee. "
        "Daarnaast is opvallend dat hoe langer de naam, hoe hoger de overlevingskans. "
        "Dit kan mogelijk verklaard worden doordat rijkere mensen vaak langere titels of meerdere namen hadden, waardoor deze waarschijnlijk in de eerste klas zitten."
    )

    st.markdown("---")
                    
    st.markdown(
        "<h3 style='color:#4CAF50;'>Crosstab vs Survived</h3><p>(0 = Not Survived, 1 = Survived)</p>",
        unsafe_allow_html=True
    )

    kolommen = [
    'Sex',
    'Pclass',
    'Embarked',
    'AgeGroup',
    'FamilySizeGroup',
    'Title',
    'IsAlone',
    'NameLength'
    ]

    candidate_cols = [c for c in kolommen if c in df.columns]
    default_dims = [c for c in ['Sex', 'Pclass'] if c in candidate_cols]
    dims = st.multiselect("Kies 1–3 kolommen", options=candidate_cols, default=default_dims, max_selections=3)

    col1, col2 = st.columns(2)
    with col1:
        min_n = st.number_input("Minimale groepsgrootte", min_value=1, value=5, step=1)
    with col2:
        show_counts = st.checkbox("Toon aantallen", value=True)

    if not dims:
        st.info("Kies minimaal één kolom.")
        st.stop()

    group_sizes = df.groupby(dims).size()
    valid_idx = group_sizes[group_sizes >= min_n].index

    ct = pd.crosstab([df[d] for d in dims], df['Survived'], normalize='index')

    ct_filtered = ct.loc[valid_idx] if len(dims) > 1 else ct[ct.index.isin(valid_idx)]

    st.caption("Crosstab")
    st.dataframe(ct_filtered)

    if show_counts:
        counts = pd.crosstab([df[d] for d in dims], df['Survived'])
        counts_filtered = counts.loc[valid_idx] if len(dims) > 1 else counts[counts.index.isin(valid_idx)]
        st.caption("Aantallen per groep")
        st.dataframe(counts_filtered)

    def flatten_index(idx):
        from pandas import MultiIndex
        if isinstance(idx, pd.MultiIndex):
            return idx.map(lambda t: " | ".join(map(str, t)))
        return idx.astype(str)

    plot_df = ct_filtered.reset_index()
    plot_df['Group'] = flatten_index(plot_df.set_index(dims).index if len(dims) > 1 else plot_df[dims[0]])

    if 0 in plot_df.columns: plot_df = plot_df.rename(columns={0: "Not Survived"})
    if 1 in plot_df.columns: plot_df = plot_df.rename(columns={1: "Survived"})

    surv_cols = [c for c in ["Not Survived", "Survived"] if c in plot_df.columns]
    if len(surv_cols) == 2:
        melt = plot_df.melt(id_vars=['Group'], value_vars=surv_cols,
                            var_name='Status', value_name='Percentage')
        fig = px.bar(melt, x='Group', y='Percentage', color='Status',
                    barmode='stack', title='Overlevingskans per gekozen groep', text='Percentage')
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(texttemplate="%{y:.0%}", textposition="inside")
        fig.update_layout(xaxis_title="Groep", yaxis_title="Percentage")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.imshow(ct_filtered, text_auto=True, aspect='auto', title='Crosstab heatmap')
        st.plotly_chart(fig, use_container_width=True)

    excluded = group_sizes[group_sizes < min_n]
    if not excluded.empty:
        st.caption("Uitgesloten groepen (< minimale groepsgrootte):")
        st.write(excluded)

    st.info(
        "Uit de crosstabs blijkt dat jonge mannen nog een redelijke overlevingskans hadden "
        "wanneer zij in de eerste of tweede klas zaten. "
        "Ook vrouwen uit de derde klas die opstapten in Queenstown (Q) of Cherbourg (C) "
        "hadden een opvallend hoge overlevingskans."
    )

    st.markdown("---")


with tab3:
    st.header("Model")
    st.write(
        "Het gebruikte model, de regels en de behaalde Kaggle-score."
    )
    st.markdown("---")

    test = pd.read_csv('bestanden/test.csv')
    st.markdown("""
    Het gekozen model is uitgebreid ten opzichte van het vorige model:

    - Vrouwen uit de 1e en 2e klas overleven het  
    - **Nieuw:** Vrouwen met 'Embarked' C of Q  uit de 3e klas overleven het  
    - Mannen uit de 1e of 2e klas jonger dan 12 jaar overleven het
    """)
    
    st.write("""De regels worden toegepast op een test dataset van kaggle.""")
    st.write("Test-data eerste paar regels:")
    st.dataframe(test.head())
    
    def apply_rules(df: pd.DataFrame) -> pd.Series:
        return np.where(
            # Vrouwen uit 1e of 2e klas
            ((df['Sex'] == 'female') & (df['Pclass'].isin([1, 2]))),
            1,
            np.where(
                # Vrouwen uit 3e klas met Embarked Q of C
                ((df['Sex'] == 'female') & (df['Pclass'] == 3) & (df['Embarked'].isin(['Q', 'C']))),
                1,
                np.where(
                    # Mannen uit 1e of 2e klas onder de 12 jaar
                    ((df['Sex'] == 'male') & (df['Pclass'].isin([1, 2])) & (df['Age'] < 12)),
                    1,
                    0
                )
            )
        ).astype(int)

    test_pred = apply_rules(test)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_pred
    })

    csv_buf = StringIO()
    submission.to_csv(csv_buf, index=False)

    if st.download_button(
        label="Download submission.csv",
        data=csv_buf.getvalue(),
        file_name="submission.csv",
        mime="text/csv"
        
    ): 
        st.success("submission.csv is succesvol gedownload!")

    st.markdown("Met deze submission wordt een Kaggle-score van **0.784** behaald.")
    st.markdown("---")
    

    

