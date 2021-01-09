import streamlit as st
import pandas as pd
import numpy as np
import shap

import matplotlib.pyplot as plt
import altair as alt
import requests as r
import xgboost as xg 
import pickle


#@st.cache
def load_model():
    #load model for the data
    return pickle.load(open("xgb_r_model_decks_with_meta.pickle.dat", "rb"))

@st.cache

def get_data():
    
    data = pd.read_csv('top_hearthstone_decks_20200221.csv')

    ## Cards info
    cat = ['card_0', 'card_1', 'card_2', 'card_3', 'card_4', 'card_5', 'card_6',
           'card_7', 'card_8', 'card_9', 'card_10', 'card_11', 'card_12',
           'card_13', 'card_14', 'card_15', 'card_16', 'card_17', 'card_18',
           'card_19', 'card_20', 'card_21', 'card_22', 'card_23', 'card_24',
           'card_25', 'card_26', 'card_27', 'card_28', 'card_29', 'hero']

    # get all categories

    categories = np.array(data[cat].values.reshape(736*31))

    # drop na
    categories = categories[~pd.isnull(categories)]

    # drop duplicates
    categories = np.unique(categories)

    cards_df = pd.DataFrame(columns=categories, index = data.index)

    for i in data.index:
        #print('deck ', i,' from ', len(data))
        cards_df.loc[i,data.loc[i,cat].dropna().values] = 1

    cards_df.fillna(0, inplace = True)

    response = r.get('https://api.hearthstonejson.com/v1/68600/enUS/cards.json'
                     ,verify = False)

    cards_info = pd.read_json(response.content)

    # some adjustments for the data

    cards_info['neutral_class'] = cards_info.cardClass.apply(lambda x: 1 if x == 'NEUTRAL' else 0)

    cards_info = cards_info.astype({'name':'category', 'type':'category'})

    # putting some mechanics in separate columns

    mechanics = ['BATTLECRY','DEATHRATTLE','TAUNT','AURA','SECRET','DISCOVER','RUSH']

    for mech in mechanics:
        cards_info[mech] = cards_info.mechanics.dropna().apply(lambda x: np.isin(x,mech)[0])

    cards_info[mechanics] = cards_info[mechanics].fillna(False)*1

    cards_info = cards_info.loc[cards_info.name.isin(cards_df.columns)
                                ,['cardClass','attack', 'neutral_class', 'cost', 'health',
                                   'name', 'type','spellDamage', 'BATTLECRY','DEATHRATTLE','TAUNT','AURA'
                                  ,'SECRET','DISCOVER','RUSH']]

    # some cards will be duplicated as they may have different attributes in different circumstanses. We only keep one variant of a card

    cards_info = cards_info.loc[~cards_info.duplicated(subset = ['name'], keep='first')]

    return cards_df, cards_info, data

cards_df, cards_info, data = get_data()

## ******************** Displaying the data ********************



hero = st.sidebar.selectbox('Hero',cards_info['cardClass'].unique()[1:])

selection = alt.selection_multi(fields=['hero'], bind='legend')

chart = alt.Chart(data).mark_area(
    opacity=0.6,
    interpolate='step'
).encode(
    alt.X('wr:Q', bin=alt.Bin(maxbins=100), title='win rate'),
    alt.Y('count()', stack=None, title='number of decks'),
    alt.Color('hero:N'),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))

).add_selection(
    selection
)

# Filter cards for a specific class

cards_list = cards_info[(cards_info['cardClass'] == hero)|(cards_info['cardClass'] == 'NEUTRAL')].name.values

st.altair_chart(chart)

selected_deck = pd.DataFrame(index = cards_df.columns, columns=[0]).fillna(0)

selected_cards = st.sidebar.multiselect('Select cards: ',cards_list)

if len(selected_cards)<15:
    #st.stop()
    st.warning('Please select at least 15 cards')

selected_deck.loc[selected_cards] = 1

#st.dataframe(selected_deck)

## ************************************************************
##******************** Composing a deck ********************

def add_meta_to_deck(deck):
    
    card_mask = deck[0]
    
    # number of cards with BATTLECRY	DEATHRATTLE	TAUNT	AURA	SECRET	DISCOVER	RUSH
    meta_data = cards_info.loc[cards_info.name.isin(card_mask[card_mask.values>0].index),['BATTLECRY','DEATHRATTLE','TAUNT',
                                                    'AURA','SECRET','DISCOVER','RUSH']].sum()

    
    #costs of cards
    costs = pd.DataFrame(index=[i for i in range(11)])
    costs[0] = cards_info.loc[cards_info.name.isin(card_mask[card_mask>0].index),'cost'].value_counts()
    costs.index = ['cost_'+str(int(i)) for i in costs.index]

    costs[0] = costs[0].fillna(0)

    meta_data = meta_data.append(costs)
    
    

    #types of cards
    card_types = pd.DataFrame(cards_info.loc[cards_info.name.isin(card_mask[card_mask.values>0].index),'type'].value_counts(), columns=[0])
    
    meta_data = meta_data.append(card_types)
    #st.text(meta_data.name)
    #st.text(.name)
    return card_mask.append(meta_data)


selected_deck_meta = add_meta_to_deck(selected_deck)

st.dataframe(selected_deck_meta[selected_deck_meta[0]>0])

model = load_model()

test_X = pd.DataFrame(selected_deck_meta).T
pred = model.predict(test_X)
       
if len(selected_cards)>=15:
    
    st.text('Expected win rate of the deck: '+ str(round(pred[0]*100,2))+'%')
    
    # SHAP features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_X)

    #shap.summary_plot(shap_values, test_X, max_display=100)
    shap.force_plot(explainer.expected_value, shap_values[0], test_X.iloc[0],link='logit', matplotlib=True, figsize=(12,3))

    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
