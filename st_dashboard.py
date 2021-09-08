import streamlit as st
import dill
import pandas as pd
import plotly.express as px
from datetime import date
import statsmodels

with open('DWH\compiled-sentiment-history.pkd', 'rb') as f: 
    df_compiled = dill.load(f)
    df_compiled.drop_duplicates(inplace=True)
dates = list({idx[1] for idx in df_compiled.index})
dates = sorted(dates, key=lambda dt: (str(dt).split('-')))

# date_ = '2021-06-01'


st.title('The Data Incubator Capstone Project')
st.subheader('*Title*: **Wallstreetbets Gossip vs. Market Price Action**')
st.subheader('*Created by*: Amir A. Taghavey - Summer, 2021')
st.markdown('*Email*: a [dot] taghavey @ gmail [dot] com')

''' '''
st.markdown(
    'This App was developed as main deliverable of thecapstone project requirement of [**the Data Incubator**](https://www.thedataincubator.com/) fellowship program.')

st.sidebar.title('Options Dashboard:')
page = st.sidebar.selectbox('Select field:',
                     (
                         'Synopsis',
                         'App structure',
                         'VIZ: Reddit hot_10 vs. time',
                         'VIZ: Gossip vs. Action', 
                         'ML analysis summary',
                         'Acknowledgments')
                         , 0)

if page == 'Synopsis':
    st.markdown(
        '''
        **Background**: The short-squeeze of GameStop and AMC stocks in early 2021 was impacted in great part by the massive-scale coordinated action of the subreddit ***wallstreetbets*** ants army of retail investors.
        Many of the early ants realized remarkable gains on their investment enabling them to payoff their student loans or home mortgages at the demise of a few hedge funds such as the London-based White Square Capital.
        These events motivated new swarms of retail investors to join in the movement with their hard-earned savings, and for many this game has offered its ugly face!

        **Objective**: Motivated by the story above, this project aimed at finding an objective answer to one question: ***Is safety in being a part of the herd when it comes to navigating the US Stock Market?***

        **Methods**: To achieve this, I (i) scanned popular social media platforms to identify and characterize how the retail investors percieved the market performance for the most frequently talked about stocks on New York Stock Exchange before each trading session and (ii) compiled the actual market action data at the end of each trading session on a daily basis over the time period of 6/1/2021-9/1/2021, and performed an extensive amount of analysis to extract possible underlying correlartions.

        **Summary**: NO correlation (and hence NO basis for meaningful predictions) was found betweem the market price action and any of the prior (i) PRE-market gossip / sentiment, (ii) stock price action, or (iii) stock options activity from the previous trading session. 

        **Conclusion**: Moral of the story, objectively and in a nutshell, is that  ***No evidence was found to support ANY consistent forward temporal correlation bwteen market gossip and price action!***
        '''
    )
elif page == 'App structure':
    st.markdown(
        ''' 
        App Structure:
        \n
            A. *reddit's PRE-market hot_20* (9:00 AM ET), the 20 most talked about NYSE stocks are identified 
            B. recent posts from *stocktwits* and *twitter* APIs for the hot_20 list of the day are compiled
            C. vader sentiment intensity analyzer is implemented to extract investor sentiment from compiled text
            D. price action data are collected from *yahoo_fin* API at the close of market (4:00 PM ET)
            E. investor sentiment - market performance data are analyzed, modeled, and visualized
            
        ''') 
    img = './CodeStructure.png'
    st.image(img, clamp=True,
        caption='Schematic of the logical code structure and inter-connections between modules \
            (i) compiling market talk data from social media platforms, \
            (ii) performing sentiment intensity analysis, \
            (iii) gathering financial data, and \
             (iv) conducting data analytics on compiled market gossip - price action data.')

elif page == 'ML analysis summary':
    st.subheader('**Machine Learning Correlation Analysis**')
    st.markdown('''
        
        \n
        ***Summary:*** An extensive correlation analysis study of the compiled data was conducted 
        with the *objective* to find underlying forward temporal correlations (if any) between 
        (a) post-market price action and (b.1) pre-market sentiment nalysis data, (b.2) pre-market 
        stock options activity data (e.g., contract volume, change in open interest, change in percent ITM / OTM, etc.), 
        and/or (b.3) previous trading session post-market price action data for reddit's hot stock list. 
        \n
        ***Approach***: Target (i.e. lable) was to predict the change in stock price, $$\Delta$$P. 
        Price change was defined as price quote at market close less price quote at market open normalized to
        price quote at market open for a given ticker on reddit hot list. Two types of approaches were implemented
        to model $$\Delta$$P: **A. Regressive Approach**, and **B. Binary Classification Approach**. 
        In the latter approach, price action signal was reduced to upward / downward trends.
        \n
        ***Transformations***: All quantitative features were scaled using standard scaler, and dimensionality 
        reduction was carried out using TrauncatedSVD method. 
        \n
        ***Modeling***: Cross validation score was used to compare modeling performance of the tested models.
        Model comparisons among regressors and classifiers were done separately using $$r^{2}$$ and accuracy 
        metrics, respectively. 
        \n
        Models implemented include:
        \n
        | Model             | Regression | Classification |
        | :---              | :--------: | :------------: |
        | Linear Regression | ✔ |  | 
        | Logistic Regression | | ✔ |
        | Ridge with cross-validation | ✔ | ✔ |
        | Decision Tree | ✔  | ✔ |
        | Random Forest  | ✔  | ✔ |
        | K-Nearest-Neighbors  | ✔  | ✔ |
        | Support Vector Machine  | ✔  | ✔ |
        | Multi-layer Perceptron Network  | ✔  | ✔ |
        \n
        . 
        \n
        ***Results***: All regressors returned an $$r^{2}$$-value equal to zero (0) consistent with no detectable correlation
        between any of (i) sentiment, (ii) stock options, or (iii) previous-day stock data and the response
        variable (i.e. $$\Delta$$P). This was further corroborated with the slighly higher than the null-model 
        classification accuracy score yielded by the KNN classifier of 0.54 (versus 0.53 classification 
        accuracy corresponding to the null hypothesis).
        The modeling results could extract no correlation between (signal) price action data for the 
        reddit hotlist and the sentiment extracted from the market talks, option activities or prior 
        trading-session data. 

    ''')

elif page == 'Acknowledgments':
    st.markdown(''' 
    - Reddit hotlist sentiment intensity analysis in this project was done by implementing an exising 
    [reddit-sentiment_analyis](https://github.com/asad70/reddit-sentiment-analysis) github repository 
    developed by [**asad70**](https://github.com/asad70). It was modified to expend search scope 
    to additional financial sub-reddits, provide human-guided training to Vader Sentiment Intensity 
    Analyzer, and to fit the required i/o structure of this project.
    - I would like to thank and acknowledge Dr. [Robert Schroll](robert@thedataincubator.com), 
    my instructor and TDI capstone project advisor, for the instrumental feedback I received from him
    during the design, development and execution of this project. 
    ''')
elif page == 'VIZ: Gossip vs. Action':
    trendline_on = st.sidebar.checkbox('add linear trendline:', False)
    date_idx = st.sidebar.slider('Select date index:', 
                                min_value=0,
                                max_value=len(dates)-1,
                                value=0)
    date_ = dates[date_idx]
    df = df_compiled.loc[(slice(None), date_),:]
    df.sort_values('counts', ascending=False, inplace=True)
    df.reset_index(inplace=True)


    # plt = sentiment_visualizer_date(c_df,'2021-06-01')
    plt=px.scatter(df, 
                    x='bull_bear_ratio',
                    y='change_sn', 
                    color='neutral', 
                    size='counts', #text='ticker', 
                    size_max=20, 
                    color_continuous_scale=px.colors.sequential.BuPu_r, 
                    hover_data=['ticker', 'volume'],
                    labels={'bull_bear_ratio': 'Investor Bullishness [-]',
                            'change_sn': 'Price Change [-]'},
                    trendline='ols' if trendline_on else None,
                    title=f"As of {date.strftime(date_, r'%B, %d %Y')}:"
                )
    plt.update_layout(plot_bgcolor='white', # #ceced0
                    title_font={'size':16, 'family':'Arial Black'},
                    yaxis={'showgrid':False, 'zeroline':False, 'linecolor': 'black', 
                            'zerolinecolor': 'grey', 'tickfont':{'size':12}, 
                            'titlefont':{'size':14, 'family':'Arial Black'},
                            'range':[-0.2,0.2]}, 
                    xaxis={'showgrid':False, 'zeroline':False, 'linecolor': 'black',
                            'tickfont':{'size':12}, 'titlefont':{'size':14, 'family':'Arial Black'},
                            'range':[.75,1.75]},
                    height=600, width=700, #'ylorrd'
                    coloraxis_colorbar={'title':"Neutrality",
                                        'tickvals': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] ,
                                        'tick0': 0.4,
                                        # 'cmin':0.5,
                                        # 'cmax': 1.0,
                                        #'tickvals':[5,6,7,8,9], 'ticktext': ['0.1M', '1M', '10M', '100M', '1B']
                                        },
                    hovermode="x unified"
                    )
    plt.update_traces(textposition='top center', 
                    textfont={'size':10, 'color':'grey'},
                    marker={'line':{'color':'#ceced0'}},
                    #hovertemplate=None,
                     )


    st.plotly_chart(plt, use_container_width=True)

    st.subheader('Sentiment')
    st.dataframe(df[['ticker', 'bearish', 'bullish', 
                        'neutral', 'bull_bear_ratio', 
                        'change_sn', 'volume']])
    
elif page == 'VIZ: Reddit hot_10 vs. time':
    st.subheader('All-time (since the Memorial Day weekend!) HOT-10 stocks on Reddit:')
    hot_10_inds = df_compiled.reset_index().groupby(by='ticker') \
                    .count()[['date']].sort_values('date', ascending=False)[:10].index
    df_ = df_compiled.reset_index()
    hot10_counts = df_[df_.ticker.isin(hot_10_inds)] \
                    .groupby('ticker') \
                    .sum()[['counts']] \
                    .reindex(hot_10_inds) \
                    .reset_index()  

    fig = px.pie(hot10_counts, values='counts', names='ticker', hole=0.3,
                    color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    
    hot10 = [f'{i+1}. {ticker}' for i, ticker in enumerate(hot_10_inds)]

    picked_hot = st.sidebar.selectbox('choose ticker to plot:', options=hot10, index=0)
    picked_hot = picked_hot.split(' ')[1]
    
    st.markdown(f'Bar chart of daily intra-session change in stock price for **${picked_hot}**:')
    
    df = df_compiled.loc[picked_hot].drop(columns=['counts'])
    
    plt = px.bar(df, y='change_sn', text='volume', color='bull_bear_ratio',
                 color_continuous_scale=px.colors.sequential.RdBu_r)
    plt.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    plt.update_layout(uniformtext_minsize=8)
    plt.update_layout(xaxis_tickangle=-45,
                        yaxis={'showgrid':False,
                                'title': 'session change [-]', 
                                'range':[-0.1, 0.1]},
                        coloraxis_colorbar={'title':"Investor\nBullishness",
                                            'tickmode': 'array',
                                            'tickvals': [0.8, 0.9, 1, 1.1, 1.2],
                                            'tick0': 0.8,})
    st.plotly_chart(plt, use_container_width=True)

    st.dataframe(df)
    